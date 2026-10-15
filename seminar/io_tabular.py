"""Read any country/year data file into one flat DataFrame.

Output shape is always the same, with a normal RangeIndex and no MultiIndex:

       iso3  year  trade_openness
    0   USA  1990           20.55
    1   USA  1991           20.31
    2   DEU  1990           52.10

Both common file layouts are accepted and detected automatically:

  LONG   country,year,value          one row per country-year
  WIDE   country,1990,1991,1992,...  one column per year

Files with no year at all are read as constants: year is left empty and the
value applies to every year in the study period (useful for a classification
file that is a single snapshot).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import countries

COUNTRY_COLUMN_CANDIDATES = (
    "iso3", "iso_3", "iso3c", "iso", "countryiso3code", "country_code",
    "countrycode", "code", "country", "country_name", "countryname",
    "country_txt", "economy", "nation", "state", "name", "entity",
)
YEAR_COLUMN_CANDIDATES = ("year", "iyear", "date", "time", "period", "yr")
VALUE_COLUMN_CANDIDATES = ("value", "val", "obs_value", "amount", "data")

_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")


class DataFileError(Exception):
    """Raised with a specific, fixable message when a file cannot be used."""


# --------------------------------------------------------------------------
# raw reading
# --------------------------------------------------------------------------

def _looks_like_header(cells: List[str]) -> bool:
    """A header row names a country column or holds several year labels."""
    lowered = [str(c).strip().lower() for c in cells if str(c).strip()]
    if not lowered:
        return False
    if any(c in COUNTRY_COLUMN_CANDIDATES for c in lowered):
        return True
    year_like = sum(1 for c in lowered if _YEAR_RE.fullmatch(c.split()[0]))
    return year_like >= 3


def read_raw(path: Path) -> pd.DataFrame:
    """Read a csv/xlsx into a DataFrame, skipping any preamble rows.

    World Bank and UNDP downloads put several metadata lines above the real
    header; this finds the real one instead of failing.
    """
    path = Path(path)
    if not path.exists():
        raise DataFileError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        probe = pd.read_excel(path, header=None, nrows=12)
        header_row = _detect_header_row([list(probe.iloc[i].values)
                                         for i in range(len(probe))])
        table = pd.read_excel(path, header=header_row)
    elif suffix in (".csv", ".txt", ".tsv"):
        separator = _detect_separator(path, suffix)
        header_row = _detect_header_row(_probe_lines(path, separator))
        table = pd.read_csv(path, header=header_row, sep=separator,
                            encoding="utf-8-sig", skip_blank_lines=True)
    else:
        raise DataFileError(f"unsupported file type '{suffix}': {path.name}")

    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if table.empty:
        raise DataFileError(f"{path.name} has no data rows")
    table.columns = [str(c).strip() for c in table.columns]
    return table


def _detect_separator(path: Path, suffix: str) -> str:
    """Comma unless the file plainly uses tabs or semicolons."""
    if suffix == ".tsv":
        return "\t"
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        sample = "".join(handle.readline() for _ in range(12))
    counts = {separator: sample.count(separator) for separator in (",", "\t", ";")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _probe_lines(path: Path, separator: str, how_many: int = 12) -> List[List[str]]:
    """The first rows as lists of strings, tolerating ragged preamble lines."""
    import csv as csv_module

    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace",
              newline="") as handle:
        for row in csv_module.reader(handle, delimiter=separator):
            rows.append(row)
            if len(rows) >= how_many:
                break
    return rows


def _detect_header_row(rows: List[List[str]]) -> int:
    for index, row in enumerate(rows):
        if _looks_like_header(list(row)):
            return index
    return 0


# --------------------------------------------------------------------------
# column identification
# --------------------------------------------------------------------------

def _pick_column(columns: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    lowered = {c.strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for candidate in candidates:               # substring fallback
        for low, original in lowered.items():
            if candidate in low:
                return original
    return None


def _year_columns(columns: List[str]) -> Dict[str, int]:
    """Columns that name a year: '1990', '1990 [YR1990]', 'y1990'."""
    found = {}
    for column in columns:
        text = str(column).strip()
        match = _YEAR_RE.search(text)
        if match and len(re.sub(r"[^0-9]", "", text)) <= 8:
            found[column] = int(match.group(1))
    return found


def _to_number(series: pd.Series, scale: float = 1.0) -> pd.Series:
    """Coerce to float, tolerating thousands separators and '..' for missing."""
    if series.dtype.kind in "if":
        numeric = series.astype(float)
    else:
        text = (series.astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.strip())
        text = text.replace({"..": np.nan, "": np.nan, "-": np.nan,
                             "n/a": np.nan, "N/A": np.nan, "nan": np.nan,
                             "None": np.nan})
        numeric = pd.to_numeric(text, errors="coerce")
    return numeric * scale


# --------------------------------------------------------------------------
# the main entry point
# --------------------------------------------------------------------------

def load_table(path: Path, name: str,
               spec: Optional[object] = None,
               numeric: bool = True) -> pd.DataFrame:
    """Load one data file as DataFrame[iso3, year, <name>].

    `name` becomes the value column name. `spec` may supply `column`,
    `country_column`, `year_column` and `scale`; anything absent or None is
    auto-detected. `numeric=False` keeps the values as text, for categorical
    classifications such as the World Bank income group.
    """
    path = Path(path)
    raw = read_raw(path)
    columns = list(raw.columns)
    source = path.name
    value_column = getattr(spec, "column", None)
    spec_country = getattr(spec, "country_column", None)
    spec_year = getattr(spec, "year_column", None)

    country_column = (spec_country if spec_country
                      else _pick_column(columns, COUNTRY_COLUMN_CANDIDATES))
    if country_column is None:
        raise DataFileError(
            f"{source}: cannot find a country column. Expected one of "
            f"{COUNTRY_COLUMN_CANDIDATES[:6]}..., got {columns[:8]}. "
            f"Set country_column= on the spec in config.py."
        )
    if country_column not in raw.columns:
        raise DataFileError(
            f"{source}: country_column='{country_column}' is not in the file. "
            f"Available columns: {columns}"
        )

    year_column = spec_year
    if year_column is None:
        year_column = _pick_column(
            [c for c in columns if c != country_column], YEAR_COLUMN_CANDIDATES
        )
    if year_column is not None and year_column not in raw.columns:
        raise DataFileError(
            f"{source}: year_column='{year_column}' is not in the file. "
            f"Available columns: {columns}"
        )

    scale = getattr(spec, "scale", 1.0)

    if year_column is not None:
        table = _load_long(raw, source, country_column, year_column, name,
                           value_column, scale, numeric)
    else:
        years = _year_columns([c for c in columns if c != country_column])
        if years:
            table = _load_wide(raw, country_column, years, name, scale, numeric)
        else:
            table = _load_constant(raw, source, country_column, name,
                                   value_column, scale, numeric)

    table["iso3"] = countries.resolve(table["__country__"], source=source)
    unresolved = table["iso3"].isna().sum()
    table = table.dropna(subset=["iso3"]).drop(columns="__country__")

    table = table.dropna(subset=[name])
    if table.empty:
        raise DataFileError(
            f"{source}: no usable rows after parsing "
            f"({unresolved} rows had an unrecognised country name). "
            f"Check the country and value columns."
        )

    # One value per country-year. Duplicates are averaged, and said so.
    key = ["iso3", "year"]
    duplicates = int(table.duplicated(subset=key).sum())
    if duplicates:
        if numeric:
            table = table.groupby(key, as_index=False, dropna=False)[name].mean()
            action = "averaged them"
        else:
            table = table.drop_duplicates(subset=key, keep="last")
            action = "kept the last one"
        print(f"  note: {source} had {duplicates} duplicate country-year rows; "
              f"{action}")

    table = table[["iso3", "year", name]].sort_values(key).reset_index(drop=True)
    table["year"] = table["year"].astype("Int64")
    if numeric:
        table[name] = table[name].astype(float)
    return table


def _values(series: pd.Series, scale: float, numeric: bool) -> pd.Series:
    """Numbers for regressions, trimmed text for categorical classifications."""
    if numeric:
        return _to_number(series, scale)
    text = series.astype(str).str.strip()
    return text.replace({"": np.nan, "nan": np.nan, "None": np.nan,
                         "..": np.nan})


def _load_long(raw, source, country_column, year_column, name,
               value_column, scale, numeric=True) -> pd.DataFrame:
    if value_column is None:
        value_column = _pick_column(
            [c for c in raw.columns if c not in (country_column, year_column)],
            VALUE_COLUMN_CANDIDATES,
        )
    if value_column is None:
        candidates = _value_candidates(raw, (country_column, year_column),
                                       numeric)
        if len(candidates) == 1:
            value_column = candidates[0]
        else:
            raise DataFileError(
                f"{source}: cannot tell which column holds the value. "
                f"Candidates: {candidates or list(raw.columns)}. "
                f"Set column= on the spec in config.py."
            )
    if value_column not in raw.columns:
        raise DataFileError(
            f"{source}: column='{value_column}' is not in the file. "
            f"Available columns: {list(raw.columns)}"
        )

    table = pd.DataFrame({
        "__country__": raw[country_column],
        "year": pd.to_numeric(
            raw[year_column].astype(str).str.extract(_YEAR_RE, expand=False),
            errors="coerce"),
        name: _values(raw[value_column], scale, numeric),
    })
    return table.dropna(subset=["year"])


def _load_wide(raw, country_column, years: Dict[str, int], name,
               scale, numeric=True) -> pd.DataFrame:
    melted = raw.melt(
        id_vars=[country_column], value_vars=list(years.keys()),
        var_name="__yearlabel__", value_name=name,
    )
    return pd.DataFrame({
        "__country__": melted[country_column],
        "year": melted["__yearlabel__"].map(years).astype(float),
        name: _values(melted[name], scale, numeric),
    })


def _load_constant(raw, source, country_column, name, value_column,
                   scale, numeric=True) -> pd.DataFrame:
    """No year anywhere: one value per country, constant over the period."""
    if value_column is None:
        candidates = _value_candidates(raw, (country_column,), numeric)
        if len(candidates) == 1:
            value_column = candidates[0]
        else:
            raise DataFileError(
                f"{source}: no year column found, and cannot tell which column "
                f"holds the value. Candidates: {candidates}. "
                f"Set column= on the spec in config.py."
            )
    print(f"  note: {source} has no year column; treating "
          f"'{value_column}' as constant over the whole period")
    return pd.DataFrame({
        "__country__": raw[country_column],
        "year": np.nan,
        name: _values(raw[value_column], scale, numeric),
    })


def _value_candidates(raw, exclude, numeric: bool) -> List[str]:
    """Columns that could plausibly hold the value."""
    candidates = []
    for column in raw.columns:
        if column in exclude:
            continue
        if numeric:
            if _to_number(raw[column]).notna().sum() > 0:
                candidates.append(column)
        else:
            if raw[column].notna().sum() > 0:
                candidates.append(column)
    return candidates


def load_folder(directory: Path, specs: Dict[str, object]
                ) -> Dict[str, pd.DataFrame]:
    """Load every declared file in a folder. Missing files raise clearly."""
    loaded = {}
    for name, spec in specs.items():
        path = Path(directory) / spec.file
        if not path.exists():
            raise DataFileError(
                f"'{name}' is declared in config.py but {path} does not exist. "
                f"Put the file there, or remove the entry."
            )
        loaded[name] = load_table(path, name, spec)
    return loaded


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------

def apply_transform(table: pd.DataFrame, name: str,
                    transform: Optional[str]) -> pd.DataFrame:
    """Apply a spec's transform per country, in year order."""
    if not transform:
        return table

    table = table.sort_values(["iso3", "year"]).copy()
    values = table[name]

    if transform == "log":
        if (values <= 0).any():
            n_bad = int((values <= 0).sum())
            print(f"  note: '{name}' log transform dropped {n_bad} "
                  f"non-positive values")
        table[name] = np.log(values.where(values > 0))
    elif transform == "log_rate":
        # ln(1 + x/100) for a rate in per cent; undefined at or below -100%.
        table[name] = np.log1p(values.where(values > -100) / 100.0)
    elif transform == "log1p":
        table[name] = np.log1p(values.where(values >= 0))
    elif transform == "pct_change":
        table[name] = table.groupby("iso3")[name].pct_change() * 100.0
    elif transform == "diff":
        table[name] = table.groupby("iso3")[name].diff()
    elif transform == "zscore":
        grouped = table.groupby("iso3")[name]
        table[name] = (values - grouped.transform("mean")) / grouped.transform("std")

    return table.dropna(subset=[name]).reset_index(drop=True)
