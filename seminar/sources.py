"""Downloading and parsing the external data the plan needs (Part 5):

    World Bank WDI        outcomes, controls, GNI Atlas, sectoral shares
    UNDP HDR              HDI and its three dimension indicators (C1, §4.6)
    World Bank OGHIST     the historical income-group assignment (C2, C3)
    UCDP/PRIO ACD         internal and external armed conflict (§6.4)
    GeoNames              capitals and largest cities (§5.8 flags)

Every parser returns a plain DataFrame, so 0_fetch_data.py only has to write
CSVs and the parsing can be tested without a network.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pycountry
import requests

from . import countries
from .terror import _normalise_place

WDI_API = "https://api.worldbank.org/v2/country/all/indicator/{code}"

# file name -> (indicator code, description). One CSV per indicator in
# data/worldbank/, long format country,year,value.
WDI_INDICATORS: Dict[str, Tuple[str, str]] = {
    # §6.2 outcomes
    "gdp_growth": ("NY.GDP.MKTP.KD.ZG",
                   "GDP growth (annual %), constant prices"),
    "fdi_inflows": ("BX.KLT.DINV.WD.GD.ZS",
                    "Foreign direct investment, net inflows (% of GDP)"),
    "fdi_outflows": ("BM.KLT.DINV.WD.GD.ZS",
                     "Foreign direct investment, net outflows (% of GDP)"),
    # Q3 and §6.4 controls
    "trade_openness": ("NE.TRD.GNFS.ZS", "Trade (% of GDP)"),
    "investment_rate": ("NE.GDI.TOTL.ZS",
                        "Gross capital formation (% of GDP)"),
    "inflation": ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
    "gdp_constant_usd": ("NY.GDP.MKTP.KD", "GDP (constant 2015 US$)"),
    "gdp_per_capita": ("NY.GDP.PCAP.KD", "GDP per capita (constant 2015 US$)"),
    "gdp_per_capita_growth": ("NY.GDP.PCAP.KD.ZG",
                              "GDP per capita growth (annual %)"),
    "population": ("SP.POP.TOTL", "Population, total"),
    # C2 and Figure 9
    "gni_per_capita_atlas": ("NY.GNP.PCAP.CD",
                             "GNI per capita, Atlas method (current US$)"),
    # Figure 5
    "agriculture_share": ("NV.AGR.TOTL.ZS",
                          "Agriculture, forestry, fishing, value added "
                          "(% of GDP)"),
    "industry_share": ("NV.IND.TOTL.ZS",
                       "Industry incl. construction, value added (% of GDP)"),
    "services_share": ("NV.SRV.TOTL.ZS", "Services, value added (% of GDP)"),
}

HDR_URLS = [
    ("https://hdr.undp.org/sites/default/files/2023-24_HDR/"
     "HDR23-24_Composite_indices_complete_time_series.csv", "HDR 2023-24"),
    ("https://hdr.undp.org/sites/default/files/2021-22_HDR/"
     "HDR21-22_Composite_indices_complete_time_series.csv", "HDR 2021-22"),
]
# HDR column prefix -> file name in data/undp/.
HDR_SERIES = {
    "hdi": "hdi",
    "le": "life_expectancy",
    "eys": "expected_schooling",
    "mys": "mean_schooling",
    "gnipc": "gni_per_capita_ppp",
}

OGHIST_URLS = [
    "https://datacatalogfiles.worldbank.org/ddh-published/0037712/DR0090754/"
    "OGHIST.xlsx",
]
UCDP_URLS = [
    ("https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-251-csv.zip",
     "UCDP/PRIO Armed Conflict Dataset v25.1"),
    ("https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-241-csv.zip",
     "UCDP/PRIO Armed Conflict Dataset v24.1"),
]
GEONAMES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
GEONAMES_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude",
    "longitude", "feature_class", "feature_code", "country_code", "cc2",
    "admin1", "admin2", "admin3", "admin4", "population", "elevation", "dem",
    "timezone", "modified",
]
# Populated places that are whole cities; sections (PPLX), historical,
# abandoned and destroyed places are left out of the top-ten ranking.
CITY_CODES = {"PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5", "PPLC",
              "PPLG"}


def _get(url: str, timeout: int = 120, retries: int = 3) -> requests.Response:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as error:                       # noqa: BLE001
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last_error}")


# --------------------------------------------------------------------------
# World Bank WDI
# --------------------------------------------------------------------------

def fetch_wdi(code: str, first_year: int, last_year: int) -> pd.DataFrame:
    """One indicator for every country, as DataFrame[country, year, value]."""
    parameters = {"format": "json", "per_page": 20000,
                  "date": f"{first_year}:{last_year}"}
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(WDI_API.format(code=code),
                                    params=parameters, timeout=60)
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as error:                       # noqa: BLE001
            last_error = error
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"could not download {code}: {last_error}")
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        message = payload[0] if payload else "empty response"
        raise RuntimeError(f"{code}: the API returned no data ({message})")

    rows = []
    for record in payload[1]:
        iso3 = (record.get("countryiso3code") or "").strip().upper()
        if not iso3 or iso3 in countries.AGGREGATE_CODES:
            continue
        if record.get("value") is None:
            continue
        rows.append({"country": iso3, "year": int(record["date"]),
                     "value": record["value"]})
    return pd.DataFrame(rows).sort_values(["country", "year"])


# --------------------------------------------------------------------------
# UNDP Human Development Report composite time series
# --------------------------------------------------------------------------

def parse_hdr(raw: pd.DataFrame, first_year: int,
              last_year: int) -> Dict[str, pd.DataFrame]:
    """{file name: DataFrame[country, year, value]} for HDR_SERIES.

    Only the plain `<prefix>_<year>` columns are taken, so hdi_f_1998 or
    hdi_rank_2022 never leak in; aggregate rows (ZZA.VHHD ...) are dropped.
    """
    raw = raw[raw["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}")]
    raw = raw[~raw["iso3"].isin(countries.AGGREGATE_CODES)]
    out = {}
    for prefix, file_name in HDR_SERIES.items():
        years = {}
        for column in raw.columns:
            match = re.fullmatch(rf"{prefix}_(\d{{4}})", str(column).strip())
            if match:
                years[column] = int(match.group(1))
        if not years:
            continue
        melted = raw.melt(id_vars=["iso3"], value_vars=list(years),
                          var_name="column", value_name="value")
        melted["year"] = melted["column"].map(years)
        melted = melted[melted["value"].notna()
                        & melted["year"].between(first_year, last_year)]
        out[file_name] = (melted.rename(columns={"iso3": "country"})
                          [["country", "year", "value"]]
                          .sort_values(["country", "year"])
                          .reset_index(drop=True))
    return out


def fetch_hdr(first_year: int, last_year: int
              ) -> Tuple[Dict[str, pd.DataFrame], str]:
    last_error = None
    for url, label in HDR_URLS:
        try:
            response = _get(url)
        except RuntimeError as error:
            last_error = error
            continue
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                raw = pd.read_csv(io.BytesIO(response.content),
                                  encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError as error:
                last_error = error
        else:
            continue
        if "iso3" not in raw.columns:
            continue
        parsed = parse_hdr(raw, first_year, last_year)
        if parsed.get("hdi") is not None and not parsed["hdi"].empty:
            return parsed, label
    raise RuntimeError(f"could not download the HDR table: {last_error}")


# --------------------------------------------------------------------------
# World Bank historical income classification (OGHIST)
# --------------------------------------------------------------------------

def parse_oghist(sheet: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The 'Country Analytical History' sheet (read with header=None) as
    (groups[country, year, value], thresholds[year, high_income_above]).

    The sheet holds one row per economy and one column per fiscal year; the
    row labelled 'Data for calendar year' gives the GNI year each column was
    based on - that is the year used here. Codes: L, LM, UM, H; '..' = not
    classified; trailing '*' marks a footnote and is removed.
    """
    label_column = sheet.iloc[:, 1].astype(str).str.strip().str.lower()
    year_row = label_column[label_column.str.startswith("data for calendar "
                                                        "year")].index
    if year_row.empty:
        raise ValueError("OGHIST layout not recognised: no 'Data for calendar "
                         "year' row")
    year_row = year_row[0]
    years = {}
    for position, value in sheet.iloc[year_row].items():
        try:
            year = int(float(str(value).strip()))
        except ValueError:
            continue
        if 1900 < year < 2100:
            years[position] = year

    high_row = label_column[label_column.str.startswith("high income")].index
    thresholds = []
    if not high_row.empty:
        for position, year in years.items():
            text = str(sheet.iloc[high_row[0], position])
            number = re.sub(r"[^0-9.]", "", text)
            if number:
                thresholds.append({"year": year,
                                   "high_income_above": float(number)})

    code_column = sheet.iloc[:, 0].astype(str).str.strip()
    is_country = code_column.str.fullmatch(r"[A-Z]{3}")
    rows = []
    for index in sheet.index[is_country]:
        iso3 = code_column[index]
        for position, year in years.items():
            value = str(sheet.iloc[index, position]).strip().replace("*", "")
            if value in ("L", "LM", "UM", "H"):
                rows.append({"country": iso3, "year": year, "value": value})
    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def fetch_oghist() -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_error = None
    for url in OGHIST_URLS:
        try:
            response = _get(url)
            sheet = pd.read_excel(io.BytesIO(response.content),
                                  sheet_name="Country Analytical History",
                                  header=None)
            return parse_oghist(sheet)
        except Exception as error:                       # noqa: BLE001
            last_error = error
    raise RuntimeError(f"could not download OGHIST: {last_error}")


# --------------------------------------------------------------------------
# UCDP/PRIO armed conflict
# --------------------------------------------------------------------------

def _names(field) -> List[str]:
    if not isinstance(field, str) or not field.strip():
        return []
    return [part.strip() for part in field.split(",") if part.strip()]


def _government_iso3(field) -> List[str]:
    codes = []
    for name in _names(field):
        if name.startswith("Government of "):
            iso3 = countries.to_iso3(name[len("Government of "):], "UCDP")
            if iso3:
                codes.append(iso3)
    return codes


def parse_ucdp(acd: pd.DataFrame, first_year: int,
               last_year: int) -> pd.DataFrame:
    """Country-year conflict indicators from the ACD (>= 25 battle deaths):

    conflict_internal  the country is the location of an intrastate conflict
                       (type 3) or an internationalised intrastate one (type 4)
    conflict_external  the country's government fights an interstate conflict
                       (type 2), or takes part - as a primary or secondary
                       party - in a conflict located outside its territory
    Country-years absent from the output had no such conflict (0).
    """
    acd = acd[acd["year"].between(first_year, last_year)]
    internal, external = set(), set()
    for _, row in acd.iterrows():
        year = int(row["year"])
        kind = int(row["type_of_conflict"])
        locations = {countries.to_iso3(name, "UCDP")
                     for name in _names(row["location"])} - {None}
        primary = _government_iso3(row["side_a"]) \
            + _government_iso3(row["side_b"])
        secondary = _government_iso3(row.get("side_a_2nd")) \
            + _government_iso3(row.get("side_b_2nd"))
        if kind in (3, 4):
            internal.update((iso3, year) for iso3 in locations)
        if kind == 2:
            external.update((iso3, year) for iso3 in primary)
        for iso3 in primary + secondary:
            if iso3 not in locations:
                external.add((iso3, year))
    keys = sorted(internal | external)
    return pd.DataFrame({
        "country": [k[0] for k in keys], "year": [k[1] for k in keys],
        "conflict_internal": [float(k in internal) for k in keys],
        "conflict_external": [float(k in external) for k in keys],
    })


def fetch_ucdp(first_year: int, last_year: int) -> Tuple[pd.DataFrame, str]:
    last_error = None
    for url, label in UCDP_URLS:
        try:
            response = _get(url)
            archive = zipfile.ZipFile(io.BytesIO(response.content))
            name = next(n for n in archive.namelist() if n.endswith(".csv"))
            acd = pd.read_csv(archive.open(name))
            return parse_ucdp(acd, first_year, last_year), label
        except Exception as error:                       # noqa: BLE001
            last_error = error
    raise RuntimeError(f"could not download the UCDP ACD: {last_error}")


# --------------------------------------------------------------------------
# GeoNames: capitals and largest cities
# --------------------------------------------------------------------------

def parse_geonames(raw: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Per country: its capital (PPLC) and its `top` most populous other
    cities, with every spelling GeoNames records, normalised for matching."""
    alpha2 = {c.alpha_2: c.alpha_3 for c in pycountry.countries}
    raw = raw[raw["feature_code"].isin(CITY_CODES)].copy()
    raw["iso3"] = raw["country_code"].map(alpha2)
    raw = raw.dropna(subset=["iso3"])
    raw["population"] = pd.to_numeric(raw["population"], errors="coerce")

    def spellings(row) -> str:
        names = {row["name"], row["asciiname"]}
        names.update(str(row["alternatenames"]).split(","))
        normal = {_normalise_place(n) for n in names if isinstance(n, str)}
        return "|".join(sorted(n for n in normal if n))

    keep = []
    for iso3, chunk in raw.groupby("iso3"):
        capital = chunk[chunk["feature_code"] == "PPLC"].head(1)
        others = chunk[chunk["feature_code"] != "PPLC"].nlargest(top,
                                                                  "population")
        keep.append(capital.assign(is_capital=1))
        keep.append(others.assign(is_capital=0))
    cities = pd.concat(keep, ignore_index=True)
    cities["names"] = cities.apply(spellings, axis=1)
    return cities[["iso3", "name", "latitude", "longitude", "population",
                   "feature_code", "is_capital", "names"]]


def fetch_geonames() -> pd.DataFrame:
    response = _get(GEONAMES_URL)
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    raw = pd.read_csv(archive.open("cities15000.txt"), sep="\t", header=None,
                      names=GEONAMES_COLUMNS, dtype={"country_code": str},
                      keep_default_na=False, na_values=[""], quoting=3,
                      low_memory=False)
    return parse_geonames(raw)


# --------------------------------------------------------------------------
# reading back what was saved
# --------------------------------------------------------------------------

def load_imf_advanced(path) -> pd.DataFrame:
    """data/classification/imf_advanced_economies.csv (comment lines '#')."""
    table = pd.read_csv(path, comment="#", encoding="utf-8-sig")
    table["iso3"] = table["iso3"].str.strip().str.upper()
    return table


def load_income_thresholds(path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        return None
