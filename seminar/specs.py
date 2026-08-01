"""Typed declarations of what the program takes from each data file.

Every data file you add must be declared with one of these specs in config.py.
Nothing is auto-detected unless you explicitly leave a field as None.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

# Transforms that may be applied to a variable before it enters a regression.
TRANSFORMS = (None, "log", "log1p", "pct_change", "diff", "zscore")


@dataclass
class VariableSpec:
    """Base declaration shared by metrics (dependent) and controls (independent).

    file:            file name inside its folder, e.g. "trade_openness.csv"
    column:          name of the column holding the number. None = auto-detect
                     the single remaining numeric column.
    country_column:  column holding country name or ISO3 code. None = auto-detect.
    year_column:     column holding the year. None = auto-detect, and if no year
                     column exists the file is read as wide (one column per year).
    transform:       see TRANSFORMS. Applied per country, ordered by year.
    label:           human readable name used in tables and figures.
    unit:            unit shown on chart axes, e.g. "% of GDP".
    scale:           multiply raw values by this before use (e.g. 1e-9).
    """

    file: str
    column: Optional[str] = None
    country_column: Optional[str] = None
    year_column: Optional[str] = None
    transform: Optional[str] = None
    label: str = ""
    unit: str = ""
    scale: float = 1.0

    def __post_init__(self):
        if self.transform not in TRANSFORMS:
            raise ValueError(
                f"{self.file}: transform={self.transform!r} is not one of {TRANSFORMS}"
            )


@dataclass
class MetricSpec(VariableSpec):
    """A DEPENDENT variable. Each metric gets its own independent regression.

    min_years:     a country needs at least this many usable years to be kept.
    min_countries: the whole metric is skipped if fewer countries survive.
    """

    min_years: int = 10
    min_countries: int = 15


@dataclass
class ControlSpec(VariableSpec):
    """An INDEPENDENT control variable, held fixed in every model."""


@dataclass
class FilterSpec(VariableSpec):
    """Classification data used to split countries (e.g. developed vs developing).

    rule="threshold"    -> value >= threshold gets high_label, otherwise low_label.
    rule="categorical"  -> the raw text value is mapped through `categories`.

    mode="static"       -> one fixed tag per country, computed from static_stat
                           over the study period. Recommended: the group never
                           changes, so the interaction term is interpretable.
    mode="time_varying" -> a country may change group from year to year.

    static_stat: "median" | "mean" | "first" | "last" - how a country's single
                 value is derived under mode="static".
    """

    rule: str = "threshold"
    threshold: Optional[float] = None
    high_label: str = "developed"
    low_label: str = "developing"
    categories: Dict[str, str] = field(default_factory=dict)
    mode: str = "static"
    static_stat: str = "median"

    def __post_init__(self):
        super().__post_init__()
        if self.rule not in ("threshold", "categorical"):
            raise ValueError(f"{self.file}: rule must be 'threshold' or 'categorical'")
        if self.rule == "threshold" and self.threshold is None:
            raise ValueError(f"{self.file}: rule='threshold' needs a threshold value")
        if self.rule == "categorical" and not self.categories:
            raise ValueError(f"{self.file}: rule='categorical' needs a categories map")
        if self.mode not in ("static", "time_varying"):
            raise ValueError(f"{self.file}: mode must be 'static' or 'time_varying'")
        if self.static_stat not in ("median", "mean", "first", "last"):
            raise ValueError(
                f"{self.file}: static_stat must be median, mean, first or last"
            )
