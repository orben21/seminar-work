"""Assemble the regression-ready table for ONE metric.

Each metric gets its own panel and its own regressions, so a gap in one metric
never shrinks the sample of another. The result is one flat DataFrame:

    iso3  year  trade_openness  log_killed  log_killed_lag1 ...  developed

Lags are plain `groupby('iso3').shift(k)` - no MultiIndex anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import io_tabular, terror
from .specs import ControlSpec, MetricSpec


@dataclass
class Panel:
    """Everything the regression step needs for one metric."""

    metric: str
    label: str
    unit: str
    data: pd.DataFrame               # estimation sample: complete rows only
    full: pd.DataFrame               # every country-year, before dropping
    terror_column: str
    lag_columns: List[str]
    control_columns: List[str]
    notes: List[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None

    @property
    def n_countries(self) -> int:
        return 0 if self.data.empty else self.data["iso3"].nunique()

    @property
    def n_obs(self) -> int:
        return len(self.data)

    def summary(self) -> str:
        if not self.ok:
            return f"{self.metric}: SKIPPED - {self.skipped_reason}"
        span = f"{int(self.data['year'].min())}-{int(self.data['year'].max())}"
        return (f"{self.metric}: {self.n_obs:,} country-years, "
                f"{self.n_countries} countries, {span}")


def build_panel(metric_name: str,
                metric_spec: MetricSpec,
                metric_table: pd.DataFrame,
                terror_country_year: pd.DataFrame,
                controls: Dict[str, pd.DataFrame],
                control_specs: Dict[str, ControlSpec],
                classification: pd.DataFrame,
                period: Tuple[int, int],
                terror_measure: str,
                lags: List[int],
                population: Optional[pd.DataFrame] = None) -> Panel:
    """Build the panel for one metric. Never raises for thin data - it returns
    a Panel with skipped_reason set, so one bad metric cannot kill the run."""
    notes: List[str] = []
    first_year, last_year = period
    years = list(range(first_year, last_year + 1))
    label = metric_spec.label or metric_name

    metric = metric_table[metric_table["year"].between(first_year, last_year)]
    metric = metric.dropna(subset=[metric_name])
    if metric.empty:
        return _skip(metric_name, label, metric_spec,
                     f"no observations inside {first_year}-{last_year}")

    # Countries with enough history of their own to support a within estimator.
    years_per_country = metric.groupby("iso3")[metric_name].count()
    keepers = years_per_country[years_per_country >= metric_spec.min_years].index
    dropped = len(years_per_country) - len(keepers)
    if dropped:
        notes.append(f"{dropped} countries dropped for having fewer than "
                     f"{metric_spec.min_years} years of {metric_name}")

    classified = set(classification["iso3"])
    usable = sorted(set(keepers) & classified)
    missing_class = sorted(set(keepers) - classified)
    if missing_class:
        notes.append(f"{len(missing_class)} countries dropped for having no "
                     f"classification value (e.g. {missing_class[:5]})")
    if len(usable) < metric_spec.min_countries:
        return _skip(metric_name, label, metric_spec,
                     f"only {len(usable)} usable countries, "
                     f"min_countries={metric_spec.min_countries}")

    grid = pd.MultiIndex.from_product(
        [usable, years], names=["iso3", "year"]).to_frame(index=False)
    grid["year"] = grid["year"].astype(int)      # plain int, not nullable Int64

    panel = grid.merge(metric, on=["iso3", "year"], how="left")

    for name, table in controls.items():
        panel = panel.merge(table, on=["iso3", "year"], how="left")

    panel = panel.merge(classification[["iso3", "year", "group", "developed"]],
                        on=["iso3", "year"], how="left")

    # Country-years absent from the GTD had no recorded attacks: a real zero.
    # 1993 stays NaN because the GTD lost that year entirely.
    terror_grid = terror.expand_to_grid(terror_country_year, usable, years)
    panel = panel.merge(terror_grid, on=["iso3", "year"], how="left")

    if population is not None:
        panel = _add_per_capita_measures(panel, population, notes)

    if terror_measure not in panel.columns:
        return _skip(metric_name, label, metric_spec,
                     f"TERROR_MEASURE='{terror_measure}' is not one of "
                     f"{sorted(c for c in terror.TERROR_MEASURES)}")

    lag_columns = []
    panel = panel.sort_values(["iso3", "year"])
    for lag in lags:
        column = f"{terror_measure}_lag{lag}"
        panel[column] = panel.groupby("iso3")[terror_measure].shift(lag)
        lag_columns.append(column)

    panel["terror"] = panel[terror_measure]
    panel["terror_x_developed"] = panel["terror"] * panel["developed"]

    control_columns = list(controls.keys())
    required = [metric_name, "terror", "developed"] + control_columns + lag_columns
    estimation = panel.dropna(subset=required).reset_index(drop=True)

    lost = len(panel) - len(estimation)
    if lost:
        notes.append(f"{lost:,} of {len(panel):,} country-years dropped for "
                     f"missing values ({_missing_breakdown(panel, required)})")

    if estimation.empty:
        return _skip(metric_name, label, metric_spec,
                     "no country-year has the metric, the controls and the "
                     "terrorism measure all present at once")
    if estimation["iso3"].nunique() < metric_spec.min_countries:
        return _skip(metric_name, label, metric_spec,
                     f"only {estimation['iso3'].nunique()} countries survive "
                     f"the merge, min_countries={metric_spec.min_countries}")

    if estimation["terror"].std(ddof=0) == 0:
        return _skip(metric_name, label, metric_spec,
                     f"'{terror_measure}' does not vary in this sample")

    within_variation = (estimation.groupby("iso3")["terror"]
                        .std(ddof=0).fillna(0.0) > 0).sum()
    if within_variation == 0:
        return _skip(metric_name, label, metric_spec,
                     f"'{terror_measure}' never varies within a country, so "
                     f"country fixed effects would absorb it entirely")

    groups = estimation["group"].dropna().unique()
    if len(groups) < 2:
        notes.append(f"only one group present ({groups.tolist()}), so the "
                     f"developed/developing comparison cannot be estimated")

    return Panel(metric=metric_name, label=label, unit=metric_spec.unit,
                 data=estimation, full=panel, terror_column="terror",
                 lag_columns=lag_columns, control_columns=control_columns,
                 notes=notes)


def _add_per_capita_measures(panel: pd.DataFrame, population: pd.DataFrame,
                             notes: List[str]) -> pd.DataFrame:
    """attacks_per_100k / killed_per_100k, from untransformed population.

    A control named 'population' may already be in the panel, and it may have
    been log transformed, so the raw numbers come in under their own name.
    """
    raw = population.rename(columns={"population": "__population_raw__"})
    merged = panel.merge(raw, on=["iso3", "year"], how="left")
    people = merged["__population_raw__"].where(merged["__population_raw__"] > 0)
    merged["attacks_per_100k"] = merged["n_attacks"] / people * 100_000
    merged["killed_per_100k"] = merged["n_killed"] / people * 100_000
    merged["log_attacks_per_100k"] = np.log1p(merged["attacks_per_100k"])
    merged["log_killed_per_100k"] = np.log1p(merged["killed_per_100k"])
    if people.isna().all():
        notes.append("population had no usable values, so the per-capita "
                     "terrorism measures are empty")
    return merged.drop(columns=["__population_raw__"])


def _missing_breakdown(panel: pd.DataFrame, required: List[str],
                       top: int = 3) -> str:
    counts = {column: int(panel[column].isna().sum()) for column in required}
    worst = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{name} missing {count:,}" for name, count in worst
                     if count)


def _skip(metric_name, label, spec, reason) -> Panel:
    return Panel(metric=metric_name, label=label, unit=spec.unit,
                 data=pd.DataFrame(), full=pd.DataFrame(), terror_column="terror",
                 lag_columns=[], control_columns=[], skipped_reason=reason)


def coverage_matrix(table: pd.DataFrame, name: str,
                    period: Tuple[int, int]) -> pd.DataFrame:
    """countries x years matrix of 1/0 availability, for the coverage heatmap."""
    first_year, last_year = period
    inside = table[table["year"].between(first_year, last_year)]
    if inside.empty:
        return pd.DataFrame()
    matrix = (inside.assign(present=inside[name].notna().astype(int))
              .pivot_table(index="iso3", columns="year", values="present",
                           aggfunc="max", fill_value=0))
    return matrix.reindex(columns=range(first_year, last_year + 1),
                          fill_value=0)
