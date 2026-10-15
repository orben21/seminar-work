"""§6: assemble the country-year panel the regressions run on.

Merge sequence (§6.5): qualifying events -> market-year aggregates -> TSI ->
merge outcomes and controls on (iso3, year) -> attach the fixed group label.
The result is one flat DataFrame, one row per market-year of the frozen
sample, no MultiIndex:

    iso3 year group developed TSI TSI_lag1 z_NA ... gdp_growth fdi_ratio <controls>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from . import io_tabular, paths
from .io_tabular import DataFileError
from .specs import ControlSpec, OutcomeSpec, SeriesSpec
from .tsi import COMPONENTS, add_lags

FOLDER_PATHS = {
    "worldbank": paths.WORLDBANK_DIR,
    "undp": paths.UNDP_DIR,
    "classification": paths.CLASSIFICATION_DIR,
    "conflict": paths.CONFLICT_DIR,
}

TREATMENT = "TSI"
TREATMENT_LAG = "TSI_lag1"
INTERACTION = "TSI_x_dev"
INTERACTION_LAG = "TSI_lag1_x_dev"
COMPONENT_TERMS = [f"z_{c}" for c in COMPONENTS]
# An outcome's values before exclude_mean_above is applied, kept so the
# regressions can be re-run with every market as a reported check (§8.5).
ALL_MARKETS_SUFFIX = "_all_markets"


class PanelError(Exception):
    """Raised with a fixable message when a declared series cannot be used."""


def series_path(spec) -> Path:
    return FOLDER_PATHS[spec.folder] / spec.file


def load_series(spec: SeriesSpec, name: str) -> pd.DataFrame:
    """One declared series as DataFrame[iso3, year, name], transformed and
    lagged. The lag shifts by calendar year, so a gap never pairs a value with
    the wrong year."""
    path = series_path(spec)
    if not path.exists():
        raise PanelError(f"'{name}' reads {path}, which does not exist. Run "
                         f"0_fetch_data.py or fix the declaration in config.py.")
    table = io_tabular.load_table(path, name, spec)
    table = table.dropna(subset=["year"])
    table["year"] = table["year"].astype(int)
    removed = []
    if getattr(spec, "zero_is_missing", False):
        zero = table[name] == 0
        removed = list(zip(table.loc[zero, "iso3"],
                           table.loc[zero, "year"] + spec.lag))
        table.loc[zero, name] = float("nan")
    if spec.transform:
        table = io_tabular.apply_transform(table, name, spec.transform)
    if spec.lag:
        table = table.assign(year=table["year"] + spec.lag)
    result = table[["iso3", "year", name]].copy()
    result.attrs["zeros_removed"] = removed
    return result


def load_outcome(spec: OutcomeSpec, name: str) -> pd.DataFrame:
    """An outcome as DataFrame[iso3, year, name]; several files are summed and
    a country-year missing any of them is missing (never a partial sum)."""
    parts = []
    for index, file in enumerate(spec.files):
        part = load_series(SeriesSpec(file=file, folder=spec.folder),
                           f"part{index}")
        parts.append(part)
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["iso3", "year"], how="inner")
    part_columns = [f"part{i}" for i in range(len(parts))]
    merged[name] = merged[part_columns].sum(axis=1, min_count=len(parts))
    return merged[["iso3", "year", name]].dropna(subset=[name])


@dataclass
class Panel:
    data: pd.DataFrame
    outcomes: Dict[str, OutcomeSpec]
    controls: Dict[str, ControlSpec]
    notes: List[str] = field(default_factory=list)
    # outcome -> {iso3: mean over the window} for markets left out of that
    # outcome's regressions by OutcomeSpec.exclude_mean_above
    excluded: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def control_names(self) -> List[str]:
        return list(self.controls)

    def label(self, name: str) -> str:
        if name in self.outcomes:
            return self.outcomes[name].label
        if name in self.controls:
            return self.controls[name].label or name
        return name


def build_panel(tsi_table: pd.DataFrame, sample: pd.DataFrame,
                outcome_specs: Dict[str, OutcomeSpec],
                control_specs: Dict[str, ControlSpec],
                window: Tuple[int, int]) -> Panel:
    """Everything the regressions need, one row per market-year."""
    notes: List[str] = []
    first, last = window

    data = add_lags(tsi_table, [TREATMENT] + COMPONENT_TERMS)
    # The classified sample is the one authority for the group label.
    labels = ["country", "group", "developed", "borderline"]
    data = data.drop(columns=[c for c in labels if c in data.columns])
    data = data.merge(sample[["iso3"] + labels], on="iso3", how="inner")

    for name, spec in outcome_specs.items():
        data = data.merge(load_outcome(spec, name), on=["iso3", "year"],
                          how="left")
    markets = set(data["iso3"])
    for name, spec in control_specs.items():
        series = load_series(spec, name)
        hits = sorted((iso3, year) for iso3, year in
                      series.attrs.get("zeros_removed", [])
                      if iso3 in markets and first <= year <= last)
        if hits:
            where = ", ".join(f"{iso3} {year}" for iso3, year in hits)
            notes.append(f"{name}: {len(hits)} market-years recorded as "
                         f"exactly 0 treated as missing (a World Bank "
                         f"placeholder for no data): {where}")
        data = data.merge(series, on=["iso3", "year"], how="left")
        if spec.fill_missing is not None:
            data[name] = data[name].fillna(spec.fill_missing)

    data["developed"] = data["developed"].astype(float)
    data[INTERACTION] = data[TREATMENT] * data["developed"]
    data[INTERACTION_LAG] = data[TREATMENT_LAG] * data["developed"]
    data = data[data["year"].between(first, last)]
    data = data.sort_values(["iso3", "year"]).reset_index(drop=True)

    excluded: Dict[str, Dict[str, float]] = {}
    for name, spec in outcome_specs.items():
        limit = getattr(spec, "exclude_mean_above", None)
        if limit is None:
            continue
        data[name + ALL_MARKETS_SUFFIX] = data[name]
        means = data.groupby("iso3")[name].mean()
        above = means[means > limit].sort_values(ascending=False)
        if above.empty:
            continue
        data.loc[data["iso3"].isin(above.index), name] = float("nan")
        excluded[name] = {iso3: float(mean) for iso3, mean in above.items()}
        listing = ", ".join(f"{iso3} {mean:.0f}" for iso3, mean in
                            above.items())
        notes.append(f"{name}: {len(above)} markets left out of this "
                     f"outcome's regressions only - mean {first}-{last} above "
                     f"{limit:g} ({spec.unit}): {listing}. They stay in the "
                     f"sample, the TSI and every other regression; the "
                     f"regressions with them are reported as a check")

    for name in list(outcome_specs) + list(control_specs):
        share = data[name].isna().mean()
        if share > 0:
            notes.append(f"{name}: {share:.1%} of market-years missing "
                         f"({int(data[name].isna().sum())} of {len(data)})")
    return Panel(data=data, outcomes=dict(outcome_specs),
                 controls=dict(control_specs), notes=notes, excluded=excluded)


def complete_rows(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """The rows on which every column is present - the estimation sample."""
    return data.dropna(subset=columns).reset_index(drop=True)


def check_series(outcome_specs: Dict[str, OutcomeSpec],
                 control_specs: Dict[str, ControlSpec]) -> List[str]:
    """Problems with the declarations, found before anything is estimated."""
    problems = []
    for name in list(outcome_specs) + list(control_specs):
        if not str(name).isidentifier():
            problems.append(f"'{name}': the name must be a valid Python "
                            f"identifier - it becomes a regression column")
    for name, spec in outcome_specs.items():
        for file in spec.files:
            path = FOLDER_PATHS[spec.folder] / file
            if not path.exists():
                problems.append(f"outcome '{name}': {path} does not exist")
    for name, spec in control_specs.items():
        path = series_path(spec)
        if not path.exists():
            problems.append(f"control '{name}': {path} does not exist")
            continue
        try:
            load_series(spec, name)
        except (DataFileError, PanelError) as error:
            problems.append(f"control '{name}': {error}")
    return problems
