"""Splitting countries into groups - developed vs developing.

Driven entirely by the FilterSpec named in ACTIVE_FILTER in config.py. Drop a
classification file into data/filters/, declare which column and which threshold
to use, and this turns it into a `developed` dummy for the regression.

The interaction term terror x developed is what answers the paper's question, so
this module prints exactly which countries landed on each side - a wrong
threshold shows up immediately rather than three steps later.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from . import countries, io_tabular, paths
from .specs import FilterSpec


class FilterError(Exception):
    """Raised when a classification cannot be built from the declared file."""


def load_filter_values(spec: FilterSpec, name: str,
                       directory: Path = paths.FILTERS_DIR) -> pd.DataFrame:
    """Read the declared classification file as DataFrame[iso3, year, name]."""
    path = Path(directory) / spec.file
    if not path.exists():
        raise FilterError(
            f"filter '{name}' points at {path}, which does not exist. "
            f"Put the file in data/filters/ or fix `file=` in config.py."
        )
    numeric = spec.rule == "threshold"
    table = io_tabular.load_table(path, name, spec, numeric=numeric)
    if spec.transform:
        table = io_tabular.apply_transform(table, name, spec.transform)
    return table


def build_classification(spec: FilterSpec, name: str, period: Tuple[int, int],
                         directory: Path = paths.FILTERS_DIR) -> pd.DataFrame:
    """Turn a classification file into DataFrame[iso3, year, group, developed].

    mode="static"       one fixed tag per country (from static_stat over the
                        period) - the group never changes, so the interaction
                        coefficient has a clean interpretation.
    mode="time_varying" the tag may change from year to year; gaps are filled
                        forward then backward within each country.
    """
    values = load_filter_values(spec, name, directory)
    first_year, last_year = period
    years = list(range(first_year, last_year + 1))

    constant = values["year"].isna()
    in_period = values["year"].between(first_year, last_year, inclusive="both")
    values = values[constant | in_period]
    if values.empty:
        raise FilterError(
            f"filter '{name}': no rows fall inside the study period "
            f"{first_year}-{last_year}."
        )

    if spec.rule == "threshold":
        _warn_on_impossible_threshold(spec, name, values[name])

    if spec.mode == "static" or constant.all():
        table = _static_classification(values, spec, name, years)
    else:
        table = _time_varying_classification(values, spec, name, years)

    table["developed"] = (table["group"] == spec.high_label).astype(float)
    return table[["iso3", "year", "group", "developed"]]


def _static_classification(values, spec, name, years) -> pd.DataFrame:
    per_country = values.groupby("iso3")[name]
    if spec.rule == "categorical":
        summary = per_country.agg(lambda s: s.dropna().iloc[-1]
                                  if s.notna().any() else np.nan)
    elif spec.static_stat == "median":
        summary = per_country.median()
    elif spec.static_stat == "mean":
        summary = per_country.mean()
    elif spec.static_stat == "first":
        summary = per_country.first()
    else:
        summary = per_country.last()

    labels = _label(summary, spec, name)
    frame = labels.rename("group").reset_index()
    grid = pd.MultiIndex.from_product(
        [frame["iso3"].tolist(), years], names=["iso3", "year"]
    ).to_frame(index=False)
    return grid.merge(frame, on="iso3", how="left")


def _time_varying_classification(values, spec, name, years) -> pd.DataFrame:
    yearly = values.dropna(subset=["year"]).copy()
    yearly["year"] = yearly["year"].astype(int)
    yearly["group"] = _label(yearly[name], spec, name)

    grid = pd.MultiIndex.from_product(
        [sorted(yearly["iso3"].unique()), years], names=["iso3", "year"]
    ).to_frame(index=False)
    merged = grid.merge(yearly[["iso3", "year", "group"]],
                        on=["iso3", "year"], how="left")
    merged["group"] = (merged.sort_values(["iso3", "year"])
                       .groupby("iso3")["group"].ffill().bfill())
    return merged


def _label(values: pd.Series, spec: FilterSpec, name: str) -> pd.Series:
    if spec.rule == "threshold":
        numeric = pd.to_numeric(values, errors="coerce")
        labels = pd.Series(
            np.where(numeric >= spec.threshold, spec.high_label, spec.low_label),
            index=values.index, dtype="object")
        labels[numeric.isna()] = np.nan
        return labels

    normalised = values.astype(str).str.strip().str.lower()
    mapping = {str(k).strip().lower(): v for k, v in spec.categories.items()}
    unknown = sorted(set(normalised.dropna()) - set(mapping))
    if unknown:
        print(f"  warning: filter '{name}' has unmapped categories {unknown}; "
              f"those countries are dropped. Add them to `categories=` "
              f"in config.py.")
    return normalised.map(mapping)


def _warn_on_impossible_threshold(spec: FilterSpec, name: str,
                                  values: pd.Series) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        raise FilterError(
            f"filter '{name}': column holds no numbers. If it holds text "
            f"categories, set rule='categorical' in config.py."
        )
    if spec.threshold > numeric.max():
        raise FilterError(
            f"filter '{name}': threshold {spec.threshold:g} is above every "
            f"value in the file (max {numeric.max():g}), so no country would "
            f"be '{spec.high_label}'. Check the units."
        )
    if spec.threshold <= numeric.min():
        raise FilterError(
            f"filter '{name}': threshold {spec.threshold:g} is at or below "
            f"every value in the file (min {numeric.min():g}), so no country "
            f"would be '{spec.low_label}'. Check the units."
        )


def apply_country_subset(classification: pd.DataFrame,
                         subset: Optional[List[str]]) -> pd.DataFrame:
    """Restrict to an explicit ISO3 list from COUNTRY_SUBSET, if one is set."""
    if not subset:
        return classification
    wanted = {str(code).strip().upper() for code in subset}
    missing = sorted(wanted - set(classification["iso3"]))
    if missing:
        print(f"  warning: COUNTRY_SUBSET lists {missing}, which the "
              f"classification does not cover")
    return classification[classification["iso3"].isin(wanted)].copy()


def summarise(classification: pd.DataFrame, spec: FilterSpec,
              name: str, max_names: int = 12) -> str:
    """Readable summary of the split, printed by 2_check_data.py."""
    latest = (classification.dropna(subset=["group"])
              .sort_values("year").groupby("iso3").last())
    counts = latest["group"].value_counts()
    lines = [f"classification '{name}' ({spec.mode}, rule={spec.rule})"]
    if spec.rule == "threshold":
        lines.append(f"  threshold: >= {spec.threshold:g} -> {spec.high_label}")
    for group, count in counts.items():
        members = sorted(latest.index[latest["group"] == group])
        shown = ", ".join(members[:max_names])
        more = f", +{len(members) - max_names} more" if len(members) > max_names else ""
        lines.append(f"  {group:<12} {count:>4} countries: {shown}{more}")
    if counts.empty:
        lines.append("  WARNING: no country could be classified")
    return "\n".join(lines)
