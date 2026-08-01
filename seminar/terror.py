"""The Global Terrorism Database side of the project.

The GTD ships as a 104 MB xlsx with 135 columns and ~210,000 events. Parsing it
takes minutes, so it is converted once into a parquet cache; every later run
loads that in under a second.

Everything here ends up as one flat country-year DataFrame whose columns are the
terrorism measures - pick which one the regression uses with TERROR_MEASURE in
config.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import countries, paths

# The GTD lost its 1993 records; the year is genuinely absent from the database,
# it is not a year of zero attacks. Treated as missing everywhere.
GTD_MISSING_YEAR = 1993

# Only these columns are pulled out of the 135 in the file.
KEEP_COLUMNS = [
    "eventid", "iyear", "imonth", "iday",
    "country_txt", "region_txt",
    "crit1", "crit2", "crit3", "doubtterr",
    "success", "suicide", "attacktype1_txt", "targtype1_txt",
    "nkill", "nwound", "property", "propvalue", "INT_ANY", "gname",
]

NUMERIC_COLUMNS = [
    "iyear", "imonth", "iday", "crit1", "crit2", "crit3", "doubtterr",
    "success", "suicide", "nkill", "nwound", "property", "propvalue", "INT_ANY",
]

# Every measure the country-year table offers to TERROR_MEASURE.
TERROR_MEASURES = [
    "n_attacks", "n_attacks_success", "n_attacks_lethal", "n_attacks_suicide",
    "n_killed", "n_wounded", "n_casualties",
    "log_attacks", "log_killed", "log_wounded", "log_casualties",
    "terror_any", "terror_intensity",
]
COUNT_MEASURES = [
    "n_attacks", "n_attacks_success", "n_attacks_lethal", "n_attacks_suicide",
    "n_killed", "n_wounded", "n_casualties",
]


class GTDError(Exception):
    """Raised when the GTD file cannot be read or looks wrong."""


# --------------------------------------------------------------------------
# one-time conversion
# --------------------------------------------------------------------------

def convert_to_parquet(xlsx_path: Path,
                       out_path: Path = paths.GTD_EVENTS_PARQUET,
                       sheet_name: str = "Data",
                       chunk_size: int = 25_000) -> pd.DataFrame:
    """Stream the GTD xlsx into a parquet cache of just the columns we need.

    Streaming (openpyxl read_only) instead of pd.read_excel keeps memory flat;
    the whole 513 MB sheet never sits in RAM at once.
    """
    from openpyxl import load_workbook

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise GTDError(
            f"GTD file not found: {xlsx_path}\n"
            f"Check GTD_XLSX in config.py."
        )

    print(f"Reading {xlsx_path.name} ({xlsx_path.stat().st_size / 1e6:.0f} MB)")
    print("This takes a few minutes, but only has to happen once.")
    started = time.time()

    workbook = load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet_name not in workbook.sheetnames:
        raise GTDError(
            f"sheet '{sheet_name}' not found. Sheets: {workbook.sheetnames}"
        )
    worksheet = workbook[sheet_name]
    rows = worksheet.iter_rows(values_only=True)

    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    position_of = {name: i for i, name in enumerate(header)}
    missing = [c for c in KEEP_COLUMNS if c not in position_of]
    if missing:
        raise GTDError(
            f"these expected GTD columns are absent: {missing}\n"
            f"Is this really the GTD distribution file?"
        )
    keep = [c for c in KEEP_COLUMNS if c in position_of]
    positions = [position_of[c] for c in keep]

    chunks: List[pd.DataFrame] = []
    buffer: List[tuple] = []
    n_rows = 0
    for row in rows:
        width = len(row)
        buffer.append(tuple(row[p] if p < width else None for p in positions))
        n_rows += 1
        if len(buffer) >= chunk_size:
            chunks.append(pd.DataFrame(buffer, columns=keep))
            buffer = []
            print(f"  {n_rows:>7,} events   "
                  f"({time.time() - started:.0f}s)", flush=True)
    if buffer:
        chunks.append(pd.DataFrame(buffer, columns=keep))
    workbook.close()

    events = pd.concat(chunks, ignore_index=True)
    print(f"  {len(events):,} events read in {time.time() - started:.0f}s")

    for column in NUMERIC_COLUMNS:
        if column in events.columns:
            events[column] = pd.to_numeric(events[column], errors="coerce")

    events["iso3"] = countries.resolve(events["country_txt"], source="GTD")
    unresolved = events["iso3"].isna()
    if unresolved.any():
        names = sorted(events.loc[unresolved, "country_txt"].dropna().unique())
        print(f"  warning: {int(unresolved.sum()):,} events have an "
              f"unrecognised country ({len(names)} distinct names): "
              f"{names[:8]}")
        print("  add them to data/reference/country_aliases.csv to include them")
    events = events.dropna(subset=["iso3", "iyear"])
    events["iyear"] = events["iyear"].astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out_path, index=False)
    print(f"  cached -> {out_path}")
    return events


def load_events(cache_path: Path = paths.GTD_EVENTS_PARQUET) -> pd.DataFrame:
    """Load the cached events. Tells you to build the cache if it is missing."""
    if not Path(cache_path).exists():
        raise GTDError(
            f"no GTD cache at {cache_path}\n"
            f"Run 1_build_terror_cache.py once to create it."
        )
    return pd.read_parquet(cache_path)


# --------------------------------------------------------------------------
# country-year measures
# --------------------------------------------------------------------------

def filter_events(events: pd.DataFrame,
                  require_all_criteria: bool = True,
                  drop_doubted: bool = True,
                  quiet: bool = False) -> pd.DataFrame:
    """Apply the standard GTD quality filters.

    crit1/2/3 are the GTD's own inclusion criteria (political-economic-religious
    aim, intent to coerce a wider audience, outside legitimate warfare).
    doubtterr=1 flags events that may be insurgency or communal violence rather
    than terrorism. Both filters are what the literature normally applies.
    """
    before = len(events)
    filtered = events
    if require_all_criteria:
        filtered = filtered[
            (filtered["crit1"] == 1) & (filtered["crit2"] == 1)
            & (filtered["crit3"] == 1)
        ]
    if drop_doubted:
        filtered = filtered[filtered["doubtterr"] != 1]
    if not quiet:
        print(f"  quality filters kept {len(filtered):,} of {before:,} events "
              f"({len(filtered) / max(before, 1):.1%})")
    return filtered


def build_country_year(events: pd.DataFrame,
                       require_all_criteria: bool = True,
                       drop_doubted: bool = True,
                       quiet: bool = False) -> pd.DataFrame:
    """Aggregate events into one flat DataFrame[iso3, year, <measures>]."""
    filtered = filter_events(events, require_all_criteria, drop_doubted, quiet)

    work = filtered.copy()
    work["killed"] = work["nkill"].fillna(0.0)
    work["wounded"] = work["nwound"].fillna(0.0)
    work["is_lethal"] = (work["killed"] > 0).astype(int)
    work["is_success"] = (work["success"] == 1).astype(int)
    work["is_suicide"] = (work["suicide"] == 1).astype(int)

    grouped = work.groupby(["iso3", "iyear"], as_index=False).agg(
        n_attacks=("eventid", "count"),
        n_attacks_success=("is_success", "sum"),
        n_attacks_lethal=("is_lethal", "sum"),
        n_attacks_suicide=("is_suicide", "sum"),
        n_killed=("killed", "sum"),
        n_wounded=("wounded", "sum"),
    )
    grouped = grouped.rename(columns={"iyear": "year"})
    grouped["n_casualties"] = grouped["n_killed"] + grouped["n_wounded"]
    return add_derived_measures(grouped)


def add_derived_measures(table: pd.DataFrame) -> pd.DataFrame:
    """Logs, the 0/1 flag and the composite intensity index."""
    table = table.copy()
    table["log_attacks"] = np.log1p(table["n_attacks"])
    table["log_killed"] = np.log1p(table["n_killed"])
    table["log_wounded"] = np.log1p(table["n_wounded"])
    table["log_casualties"] = np.log1p(table["n_casualties"])
    table["terror_any"] = (table["n_attacks"] > 0).astype(float)
    # A single severity index: attacks and human cost weighted equally, on a
    # log scale so one catastrophic year does not dominate the panel.
    table["terror_intensity"] = (
        0.5 * table["log_attacks"] + 0.5 * table["log_casualties"]
    )
    return table


def expand_to_grid(country_year: pd.DataFrame,
                   iso3_codes: List[str],
                   years: List[int]) -> pd.DataFrame:
    """Fill in country-years with no recorded attacks as genuine zeros.

    A country absent from the GTD in a given year had no recorded attacks, which
    is information, not missing data.

    Two exceptions stay NaN so they drop out of the regressions instead of
    entering as a fake zero:
      - 1993, which the GTD lost entirely;
      - any year outside what this GTD release actually covers. The May 2022
        release stops at 2020, so asking for 2021 must not produce "no attacks
        anywhere in the world".
    """
    grid = pd.MultiIndex.from_product(
        [sorted(set(iso3_codes)), sorted(set(years))], names=["iso3", "year"]
    ).to_frame(index=False)                      # back to flat columns at once
    grid["year"] = grid["year"].astype(int)

    merged = grid.merge(country_year, on=["iso3", "year"], how="left")
    merged[COUNT_MEASURES] = merged[COUNT_MEASURES].fillna(0.0)
    merged = add_derived_measures(merged.drop(
        columns=[c for c in TERROR_MEASURES if c not in COUNT_MEASURES],
        errors="ignore"))

    covered_first = int(country_year["year"].min())
    covered_last = int(country_year["year"].max())
    outside = ~merged["year"].between(covered_first, covered_last)
    outside |= merged["year"] == GTD_MISSING_YEAR
    if outside.any():
        merged.loc[outside, TERROR_MEASURES] = np.nan
        uncovered = sorted(set(merged.loc[outside, "year"]) -
                           {GTD_MISSING_YEAR})
        if uncovered:
            print(f"  note: this GTD release covers {covered_first}-"
                  f"{covered_last}, so {uncovered} have no terrorism data and "
                  f"are excluded rather than counted as zero")

    return merged


def load_country_year(require_all_criteria: bool = True,
                      drop_doubted: bool = True,
                      cache_path: Path = paths.GTD_COUNTRY_YEAR_PARQUET,
                      events_path: Path = paths.GTD_EVENTS_PARQUET,
                      rebuild: bool = False,
                      quiet: bool = False) -> pd.DataFrame:
    """Country-year terrorism measures, cached to parquet."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)
    table = build_country_year(load_events(events_path), require_all_criteria,
                               drop_doubted, quiet)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache_path, index=False)
    return table


def describe(country_year: pd.DataFrame) -> str:
    """A short human readable summary, used by the check and build scripts."""
    years = country_year["year"]
    lines = [
        f"country-years : {len(country_year):,}",
        f"countries     : {country_year['iso3'].nunique()}",
        f"years         : {int(years.min())}-{int(years.max())}",
        f"total attacks : {int(country_year['n_attacks'].sum()):,}",
        f"total killed  : {int(country_year['n_killed'].sum()):,}",
    ]
    if GTD_MISSING_YEAR not in set(years.unique()):
        lines.append(f"1993          : absent, as expected for the GTD")
    else:
        lines.append(f"1993          : PRESENT - unexpected for the GTD, check "
                     f"the source file")
    return "\n".join("  " + line for line in lines)
