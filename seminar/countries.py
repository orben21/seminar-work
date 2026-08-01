"""Canonical country identity.

Every data source names countries differently: the GTD says "West Bank and Gaza
Strip", the World Bank says "West Bank and Gaza", the UN says "Palestine, State
of". Everything in this project is keyed on the ISO 3166-1 alpha-3 code, and
this module is the single place where a name becomes that code.

Anything that cannot be resolved is reported, never silently dropped.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

import pandas as pd
import pycountry

from . import paths

# Names that are region/income aggregates rather than countries. They appear in
# World Bank downloads and must not enter a country panel.
AGGREGATE_CODES = {
    "WLD", "ARB", "CSS", "CEB", "EAR", "EAS", "EAP", "TEA", "EMU", "ECS", "ECA",
    "TEC", "EUU", "FCS", "HPC", "HIC", "IBD", "IBT", "IDB", "IDX", "IDA", "LTE",
    "LCN", "LAC", "TLA", "LDC", "LMY", "LIC", "LMC", "MEA", "MNA", "TMN", "MIC",
    "NAC", "INX", "OED", "OSS", "PSS", "PST", "PRE", "SST", "SAS", "TSA", "SSF",
    "SSA", "TSS", "UMC", "AFE", "AFW", "AFR", "EAPEXCHN",
}

AGGREGATE_NAMES = {
    "world", "arab world", "european union", "euro area", "oecd members",
    "high income", "low income", "middle income", "lower middle income",
    "upper middle income", "low & middle income", "north america",
    "sub-saharan africa", "south asia", "east asia & pacific",
    "europe & central asia", "latin america & caribbean",
    "middle east & north africa", "least developed countries: un classification",
    "international", "unknown", "multinational", "n/a", "total",
}

# Names the pycountry database does not know, or knows differently. Includes the
# GTD's historical entities: the panel starts in 1990, which still catches the
# Soviet Union, Yugoslavia, Czechoslovakia and both Germanys.
BUILTIN_ALIASES: Dict[str, str] = {
    # --- GTD historical entities -> successor state used for the panel ---
    "soviet union": "RUS",
    "ussr": "RUS",
    "russia": "RUS",
    "russian federation": "RUS",
    "east germany (gdr)": "DEU",
    "west germany (frg)": "DEU",
    "germany, fed. rep.": "DEU",
    "czechoslovakia": "CZE",
    "czech republic": "CZE",
    "czechia": "CZE",
    "slovak republic": "SVK",
    "yugoslavia": "SRB",
    "serbia-montenegro": "SRB",
    "republic of serbia": "SRB",
    "kosovo": "XKX",
    "zaire": "COD",
    "rhodesia": "ZWE",
    "south vietnam": "VNM",
    "vietnam": "VNM",
    "north vietnam": "VNM",
    "north yemen": "YEM",
    "south yemen": "YEM",
    "yemen, rep.": "YEM",
    "new hebrides": "VUT",
    "burma": "MMR",
    "myanmar": "MMR",
    # --- everyday naming differences ---
    "united states": "USA",
    "united states of america": "USA",
    "usa": "USA",
    "united kingdom": "GBR",
    "great britain": "GBR",
    "uk": "GBR",
    "south korea": "KOR",
    "korea, rep.": "KOR",
    "korea, republic of": "KOR",
    "republic of korea": "KOR",
    "north korea": "PRK",
    "korea, dem. people's rep.": "PRK",
    "iran": "IRN",
    "iran, islamic rep.": "IRN",
    "syria": "SYR",
    "syrian arab republic": "SYR",
    "egypt": "EGY",
    "egypt, arab rep.": "EGY",
    "laos": "LAO",
    "lao pdr": "LAO",
    "taiwan": "TWN",
    "hong kong": "HKG",
    "hong kong sar, china": "HKG",
    "macau": "MAC",
    "macao": "MAC",
    "macao sar, china": "MAC",
    "china": "CHN",
    "ivory coast": "CIV",
    "cote d'ivoire": "CIV",
    "cote divoire": "CIV",
    "democratic republic of the congo": "COD",
    "congo, dem. rep.": "COD",
    "people's republic of the congo": "COG",
    "republic of the congo": "COG",
    "congo, rep.": "COG",
    "bosnia-herzegovina": "BIH",
    "bosnia and herzegovina": "BIH",
    "macedonia": "MKD",
    "north macedonia": "MKD",
    "moldova": "MDA",
    "brunei": "BRN",
    "brunei darussalam": "BRN",
    "cape verde": "CPV",
    "cabo verde": "CPV",
    "swaziland": "SWZ",
    "eswatini": "SWZ",
    "east timor": "TLS",
    "timor-leste": "TLS",
    "vatican city": "VAT",
    "holy see": "VAT",
    "west bank and gaza strip": "PSE",
    "west bank and gaza": "PSE",
    "palestine": "PSE",
    "gaza strip": "PSE",
    "bahamas": "BHS",
    "bahamas, the": "BHS",
    "gambia": "GMB",
    "gambia, the": "GMB",
    "st. kitts and nevis": "KNA",
    "saint kitts and nevis": "KNA",
    "st. lucia": "LCA",
    "saint lucia": "LCA",
    "st. vincent and the grenadines": "VCT",
    "saint vincent and the grenadines": "VCT",
    "st. martin (french part)": "MAF",
    "sint maarten (dutch part)": "SXM",
    "tanzania": "TZA",
    "venezuela": "VEN",
    "venezuela, rb": "VEN",
    "bolivia": "BOL",
    "kyrgyzstan": "KGZ",
    "kyrgyz republic": "KGZ",
    "slovakia": "SVK",
    "turkey": "TUR",
    "turkiye": "TUR",
    "wallis and futuna": "WLF",
    "falkland islands": "FLK",
    "french guiana": "GUF",
    "french polynesia": "PYF",
    "guadeloupe": "GLP",
    "martinique": "MTQ",
    "new caledonia": "NCL",
    "reunion": "REU",
    "western sahara": "ESH",
    "puerto rico": "PRI",
    "virgin islands (u.s.)": "VIR",
    "yemen": "YEM",
}

_ISO3_RE = re.compile(r"^[A-Z]{3}$")

# Populated on first use.
_lookup: Optional[Dict[str, str]] = None
_unmatched: Dict[str, set] = {}


def _normalise(name) -> str:
    """Lowercase, strip punctuation noise, collapse whitespace."""
    if name is None:
        return ""
    text = str(name).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[‘’“”]", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _load_alias_file() -> Dict[str, str]:
    """User-maintained overrides from data/reference/country_aliases.csv."""
    path = paths.COUNTRY_ALIASES_CSV
    if not path.exists():
        return {}
    table = pd.read_csv(path, encoding="utf-8-sig", comment="#")
    table.columns = [c.strip().lower() for c in table.columns]
    if "name" not in table.columns or "iso3" not in table.columns:
        raise ValueError(f"{path} must have columns: name,iso3")
    aliases = {}
    for _, row in table.iterrows():
        key = _normalise(row["name"])
        value = str(row["iso3"]).strip().upper()
        if key and _ISO3_RE.match(value):
            aliases[key] = value
    return aliases


def _build_lookup() -> Dict[str, str]:
    table: Dict[str, str] = {}

    def add(key, iso3):
        key = _normalise(key)
        if key:
            table.setdefault(key, iso3)

    for country in pycountry.countries:
        iso3 = country.alpha_3
        add(country.name, iso3)
        add(country.alpha_2, iso3)
        add(iso3, iso3)
        for attribute in ("official_name", "common_name"):
            value = getattr(country, attribute, None)
            if value:
                add(value, iso3)
        # "Bolivia, Plurinational State of" -> "Bolivia"
        if "," in country.name:
            add(country.name.split(",")[0], iso3)

    for country in pycountry.historic_countries:
        iso3 = getattr(country, "alpha_3", None)
        if iso3 and len(iso3) == 3:
            add(country.name, iso3)

    # Explicit aliases win over anything guessed above.
    table.update({_normalise(k): v for k, v in BUILTIN_ALIASES.items()})
    table.update(_load_alias_file())
    return table


def lookup_table() -> Dict[str, str]:
    global _lookup
    if _lookup is None:
        _lookup = _build_lookup()
    return _lookup


def to_iso3(name, source: str = "") -> Optional[str]:
    """Resolve one country name (or code) to ISO3, or None if unresolvable."""
    key = _normalise(name)
    if not key or key in AGGREGATE_NAMES:
        return None

    raw = str(name).strip().upper()
    if _ISO3_RE.match(raw):
        if raw in AGGREGATE_CODES:
            return None
        if raw in ("XKX", "TWN", "PSE"):  # valid here, missing from some tables
            return raw

    table = lookup_table()
    if key in table:
        iso3 = table[key]
        return None if iso3 in AGGREGATE_CODES else iso3

    # Try dropping bracketed qualifiers: "Congo (Kinshasa)" -> "Congo"
    stripped = re.sub(r"\s*\([^)]*\)", "", key).strip()
    if stripped and stripped in table:
        return table[stripped]

    _unmatched.setdefault(source or "unknown", set()).add(str(name))
    return None


def resolve(names: Iterable, source: str = "") -> pd.Series:
    """Vectorised to_iso3 over a column of country names."""
    series = pd.Series(list(names), dtype="object")
    unique = {value: to_iso3(value, source) for value in series.dropna().unique()}
    return series.map(unique)


def unmatched() -> Dict[str, set]:
    """Country names seen so far that could not be resolved, keyed by source."""
    return _unmatched


def write_unmatched_report() -> Optional[str]:
    """Write outputs/unmatched_countries.csv. Returns the path, or None."""
    if not _unmatched:
        return None
    rows = [
        {"source": source, "name": name}
        for source, names in sorted(_unmatched.items())
        for name in sorted(names)
    ]
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.UNMATCHED_COUNTRIES_CSV, index=False,
                              encoding="utf-8-sig")
    return str(paths.UNMATCHED_COUNTRIES_CSV)


def name_of(iso3: str) -> str:
    """ISO3 back to a readable country name, for charts and tables."""
    try:
        country = pycountry.countries.get(alpha_3=iso3)
        if country is not None:
            return getattr(country, "common_name", country.name)
    except (KeyError, LookupError):
        pass
    return iso3
