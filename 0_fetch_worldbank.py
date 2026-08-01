# -*- coding: utf-8 -*-
"""STEP 0 (optional) - download real metric data from the World Bank and the UN.

Your metric data was not available when this project was built, so this script
fills the folders with real, citable numbers straight from the World Bank's open
API and the UNDP's published Human Development Report tables. Everything it
writes is an ordinary CSV in the exact format the loaders expect, so you can
inspect, edit or replace any of it by hand.

    data/metrics/    trade_openness, fdi_inflows, fdi_outflows, gdp_growth,
                     market_cap
    data/controls/   gdp_per_capita, population
    data/filters/    gni_per_capita, income_group, hdi

If you are getting your metrics somewhere else (Bloomberg, an index provider,
your own spreadsheet), you do not need this script at all - just drop your files
into those folders in the same shape and declare them in config.py.

Source: World Development Indicators, https://data.worldbank.org
"""

import io
import re
import sys
import time

import pandas as pd
import requests

import config
from seminar import countries, paths

# Some networks (corporate proxies, antivirus with HTTPS scanning) present their
# own certificate. Windows trusts it, but Python ships its own certificate list
# and does not, so downloads fail with CERTIFICATE_VERIFY_FAILED. This makes
# Python use the Windows certificate store instead - it does NOT disable any
# security check.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

API = "https://api.worldbank.org/v2/country/all/indicator/{code}"

# folder -> {file name: (indicator code, what it is)}
INDICATORS = {
    "metrics": {
        "trade_openness": ("NE.TRD.GNFS.ZS",
                           "Trade (exports + imports) as % of GDP"),
        "fdi_inflows": ("BX.KLT.DINV.WD.GD.ZS",
                        "Foreign direct investment, net inflows, % of GDP"),
        "fdi_outflows": ("BM.KLT.DINV.WD.GD.ZS",
                         "Foreign direct investment, net outflows, % of GDP"),
        "gdp_growth": ("NY.GDP.PCAP.KD.ZG",
                       "GDP per capita growth, % per year"),
        "market_cap": ("CM.MKT.LCAP.GD.ZS",
                       "Market capitalisation of listed domestic companies, "
                       "% of GDP"),
    },
    "controls": {
        "gdp_per_capita": ("NY.GDP.PCAP.KD",
                           "GDP per capita, constant 2015 US$"),
        "population": ("SP.POP.TOTL", "Total population"),
    },
    "filters": {
        "gni_per_capita": ("NY.GNP.PCAP.CD",
                           "GNI per capita, Atlas method, current US$"),
    },
}

FOLDERS = {
    "metrics": paths.METRICS_DIR,
    "controls": paths.CONTROLS_DIR,
    "filters": paths.FILTERS_DIR,
}


def fetch_indicator(code: str, first_year: int, last_year: int,
                    retries: int = 3) -> pd.DataFrame:
    """One indicator for every country, as DataFrame[country, year, value]."""
    parameters = {"format": "json", "per_page": 20000,
                  "date": f"{first_year}:{last_year}"}
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(API.format(code=code), params=parameters,
                                    timeout=60)
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
            continue                                     # drop region totals
        if record.get("value") is None:
            continue
        rows.append({"country": iso3, "year": int(record["date"]),
                     "value": record["value"]})
    return pd.DataFrame(rows).sort_values(["country", "year"])


# The UNDP publishes the Human Development Index as one wide file per report.
# Newest first; the first one that downloads and parses wins.
HDI_URLS = [
    ("https://hdr.undp.org/sites/default/files/2023-24_HDR/"
     "HDR23-24_Composite_indices_complete_time_series.csv", "HDR 2023-24"),
    ("https://hdr.undp.org/sites/default/files/2021-22_HDR/"
     "HDR21-22_Composite_indices_complete_time_series.csv", "HDR 2021-22"),
]


def fetch_hdi(first_year: int, last_year: int) -> tuple:
    """HDI per country-year, as DataFrame[country, year, value] plus a source.

    The file holds about a thousand columns - HDI by sex, life expectancy,
    schooling and more. Only the plain `hdi_<year>` columns are taken, so
    `hdi_rank_2021` and `hdi_f_1995` are correctly left out.
    """
    last_error = None
    for url, label in HDI_URLS:
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
        except Exception as error:                       # noqa: BLE001
            last_error = error
            continue

        table = None
        for encoding in ("utf-8-sig", "latin-1"):        # UNDP files vary
            try:
                table = pd.read_csv(io.BytesIO(response.content),
                                    encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError as error:
                last_error = error
        if table is None or "iso3" not in table.columns:
            continue

        years = {}
        for column in table.columns:
            match = re.fullmatch(r"hdi_(\d{4})", str(column).strip())
            if match:
                years[column] = int(match.group(1))
        if not years:
            continue

        melted = table.melt(id_vars=["iso3"], value_vars=list(years),
                            var_name="column", value_name="value")
        melted["year"] = melted["column"].map(years)
        melted["country"] = melted["iso3"].astype(str).str.strip().str.upper()
        melted = melted[melted["value"].notna()]
        melted = melted[melted["year"].between(first_year, last_year)]
        # HDR aggregate rows use codes like ZZA.VHHD, which are not countries.
        melted = melted[melted["country"].str.fullmatch(r"[A-Z]{3}")]
        melted = melted[~melted["country"].isin(countries.AGGREGATE_CODES)]
        result = (melted[["country", "year", "value"]]
                  .sort_values(["country", "year"]).reset_index(drop=True))
        if not result.empty:
            return result, label

    raise RuntimeError(f"could not download the HDI table: {last_error}")


def fetch_income_groups() -> pd.DataFrame:
    """The World Bank's own income classification, as a categorical filter."""
    response = requests.get("https://api.worldbank.org/v2/country",
                            params={"format": "json", "per_page": 400},
                            timeout=60)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for record in payload[1]:
        iso3 = (record.get("id") or "").strip().upper()
        level = (record.get("incomeLevel") or {}).get("value", "")
        if not iso3 or iso3 in countries.AGGREGATE_CODES:
            continue
        if not level or level.lower() == "aggregates":
            continue
        rows.append({"country": iso3, "value": level})
    return pd.DataFrame(rows).sort_values("country")


def main() -> int:
    paths.ensure_dirs()
    first_year, last_year = config.PERIOD

    print("=" * 70)
    print(f"STEP 0  downloading World Bank data for {first_year}-{last_year}")
    print("=" * 70)
    print("Source: World Development Indicators, https://data.worldbank.org\n")

    failures = []
    for folder_key, indicators in INDICATORS.items():
        directory = FOLDERS[folder_key]
        for name, (code, description) in indicators.items():
            print(f"{name:<18} {code:<22}", end="", flush=True)
            try:
                table = fetch_indicator(code, first_year, last_year)
            except RuntimeError as error:
                print(f"FAILED - {error}")
                failures.append(name)
                continue
            path = directory / f"{name}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"{len(table):>7,} rows  "
                  f"{table['country'].nunique():>3} countries  -> "
                  f"{path.relative_to(paths.PROJECT_ROOT)}")

    print(f"{'income_group':<18} {'(classification)':<22}", end="", flush=True)
    try:
        groups = fetch_income_groups()
        path = paths.FILTERS_DIR / "income_group.csv"
        groups.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{len(groups):>7,} rows                    -> "
              f"{path.relative_to(paths.PROJECT_ROOT)}")
    except Exception as error:                           # noqa: BLE001
        print(f"FAILED - {error}")
        failures.append("income_group")

    print(f"{'hdi':<18} {'(UNDP)':<22}", end="", flush=True)
    try:
        hdi, source = fetch_hdi(first_year, last_year)
        path = paths.FILTERS_DIR / "hdi.csv"
        hdi.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{len(hdi):>7,} rows  {hdi['country'].nunique():>3} countries  "
              f"-> {path.relative_to(paths.PROJECT_ROOT)}  [{source}]")
    except Exception as error:                           # noqa: BLE001
        print(f"FAILED - {error}")
        print("  Download it by hand from https://hdr.undp.org/data-center/"
              "documentation-and-downloads")
        print("  and save it as data/filters/hdi.csv")
        failures.append("hdi")

    print("\nA note for the paper: the economic series are the World Bank's "
          "World\nDevelopment Indicators, downloaded through the public API - "
          "the indicator\ncodes above are what you cite. The HDI comes from the "
          "UNDP's Human\nDevelopment Report composite indices table.")

    if failures:
        print(f"\n{len(failures)} download(s) failed: {failures}")
        print("Re-run the script; the World Bank API is occasionally busy.")
        return 1

    print("\nDone. Next: run 2_check_data.py to see what you got.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
