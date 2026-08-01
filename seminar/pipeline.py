"""Wiring: turn the declarations in config.py into loaded, checked data.

Shared by 2_check_data.py (which only validates) and 3_run_analysis.py (which
goes on to estimate), so both see exactly the same data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import filters, io_tabular, paths
from .io_tabular import DataFileError
from .specs import ControlSpec, FilterSpec, MetricSpec


@dataclass
class LoadedData:
    metrics: Dict[str, pd.DataFrame] = field(default_factory=dict)
    controls: Dict[str, pd.DataFrame] = field(default_factory=dict)
    population_raw: Optional[pd.DataFrame] = None
    classification: pd.DataFrame = field(default_factory=pd.DataFrame)
    filter_name: str = ""
    filter_spec: Optional[FilterSpec] = None
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and bool(self.metrics)


def check_names(specs: Dict[str, object], kind: str) -> List[str]:
    """Declaration keys become column names in a formula, so they must be
    ordinary identifiers."""
    problems = []
    for name in specs:
        if not str(name).isidentifier():
            problems.append(
                f"{kind} '{name}': the name must be a valid Python identifier "
                f"(letters, digits and underscores, not starting with a digit) "
                f"because it becomes a column in the regression.")
    return problems


RESERVED = {"iso3", "year", "terror", "developed", "group",
            "terror_x_developed"}


def load_variables(specs: Dict[str, MetricSpec], directory: Path, kind: str,
                   problems: List[str]) -> Dict[str, pd.DataFrame]:
    """Load and transform every declared file in one folder."""
    loaded = {}
    for name, spec in specs.items():
        if name in RESERVED:
            problems.append(f"{kind} '{name}': that name is used internally. "
                            f"Rename it in config.py.")
            continue
        path = Path(directory) / spec.file
        try:
            table = io_tabular.load_table(path, name, spec)
            table = io_tabular.apply_transform(table, name, spec.transform)
            loaded[name] = table
        except DataFileError as error:
            problems.append(f"{kind} '{name}': {error}")
        except Exception as error:                      # noqa: BLE001
            problems.append(f"{kind} '{name}': unexpected problem reading "
                            f"{path.name} - {error}")
    return loaded


def load_population_raw(control_specs: Dict[str, ControlSpec],
                        directory: Path) -> Optional[pd.DataFrame]:
    """Untransformed population, so per-capita terrorism measures are real."""
    spec = control_specs.get("population")
    if spec is None:
        return None
    path = Path(directory) / spec.file
    if not path.exists():
        return None
    try:
        plain = ControlSpec(file=spec.file, column=spec.column,
                            country_column=spec.country_column,
                            year_column=spec.year_column, scale=spec.scale)
        return io_tabular.load_table(path, "population", plain)
    except Exception:                                   # noqa: BLE001
        return None


def load_all(metric_specs: Dict[str, MetricSpec],
             control_specs: Dict[str, ControlSpec],
             filter_specs: Dict[str, FilterSpec],
             active_filter: str,
             period: Tuple[int, int],
             country_subset: Optional[List[str]] = None,
             metrics_dir: Path = paths.METRICS_DIR,
             controls_dir: Path = paths.CONTROLS_DIR,
             filters_dir: Path = paths.FILTERS_DIR) -> LoadedData:
    """Load every declared file, or collect a clear problem for each failure."""
    data = LoadedData(filter_name=active_filter)
    data.problems.extend(check_names(metric_specs, "metric"))
    data.problems.extend(check_names(control_specs, "control"))

    if not metric_specs:
        data.problems.append(
            "no metrics are declared. Add at least one MetricSpec to METRICS "
            "in config.py, with its file in data/metrics/.")

    print("Loading metrics (dependent variables)")
    data.metrics = load_variables(metric_specs, metrics_dir, "metric",
                                  data.problems)
    for name, table in data.metrics.items():
        print(f"  {name:<22} {len(table):>7,} rows  "
              f"{table['iso3'].nunique():>4} countries  "
              f"{int(table['year'].min())}-{int(table['year'].max())}")

    print("Loading controls (independent variables)")
    data.controls = load_variables(control_specs, controls_dir, "control",
                                   data.problems)
    for name, table in data.controls.items():
        print(f"  {name:<22} {len(table):>7,} rows  "
              f"{table['iso3'].nunique():>4} countries  "
              f"{int(table['year'].min())}-{int(table['year'].max())}")
    data.population_raw = load_population_raw(control_specs, controls_dir)

    print("Building the classification (developed vs developing)")
    spec = filter_specs.get(active_filter)
    if spec is None:
        data.problems.append(
            f"ACTIVE_FILTER='{active_filter}' is not a key of FILTERS in "
            f"config.py. Available: {sorted(filter_specs)}")
        return data
    data.filter_spec = spec
    try:
        classification = filters.build_classification(
            spec, active_filter, period, filters_dir)
        data.classification = filters.apply_country_subset(
            classification, country_subset)
        print(filters.summarise(data.classification, spec, active_filter))
    except (filters.FilterError, DataFileError) as error:
        data.problems.append(f"filter '{active_filter}': {error}")
    except Exception as error:                          # noqa: BLE001
        data.problems.append(f"filter '{active_filter}': unexpected problem - "
                             f"{error}")
    return data


def report_problems(problems: List[str]) -> None:
    if not problems:
        return
    print("\n" + "!" * 70)
    print(f"{len(problems)} problem(s) found:")
    for index, problem in enumerate(problems, start=1):
        print(f"  {index}. {problem}")
    print("!" * 70)
