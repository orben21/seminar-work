"""Typed declarations of the data series the analysis reads.

Outcomes (§6.2) and controls (§6.4) are declared with these in config.py, so
the variable set of the regressions is visible in one place and checked before
anything is estimated.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

# Transforms that may be applied to a series before it enters a regression.
#   log       natural log, for levels such as GDP per capita
#   log_rate  ln(1 + x/100), for rates in per cent that can be very large,
#             such as inflation; keeps the sign and tames hyperinflation years
TRANSFORMS = (None, "log", "log_rate")

FOLDERS = ("worldbank", "undp", "classification", "conflict")


@dataclass
class SeriesSpec:
    """One country-year series read from data/<folder>/<file>.

    column:        the column holding the number (None = auto-detect).
    transform:     see TRANSFORMS; applied before the lag.
    lag:           use the value of year t-lag, per country (0 = same year).
    fill_missing:  value for country-years absent from the file. None keeps
                   them missing; 0.0 is right for event indicators such as
                   armed conflict, where absence from the dataset means none.
    zero_is_missing: treat a value of exactly 0 as missing. For series that
                   cannot truly be zero - an investment rate or a trade ratio -
                   the World Bank download sometimes carries 0 as a placeholder
                   for "no data" (Venezuela 1998-2011). Off by default.
    """

    file: str
    folder: str = "worldbank"
    column: Optional[str] = "value"
    transform: Optional[str] = None
    lag: int = 0
    fill_missing: Optional[float] = None
    label: str = ""
    unit: str = ""
    source: str = ""
    zero_is_missing: bool = False

    def __post_init__(self):
        if self.transform not in TRANSFORMS:
            raise ValueError(f"{self.file}: transform={self.transform!r} is "
                             f"not one of {TRANSFORMS}")
        if self.folder not in FOLDERS:
            raise ValueError(f"{self.file}: folder={self.folder!r} is not one "
                             f"of {FOLDERS}")
        if self.lag < 0:
            raise ValueError(f"{self.file}: lag must be 0 or positive")


@dataclass
class ControlSpec(SeriesSpec):
    """A control variable (§6.4). `precedent` records why it is in the model -
    the plan requires every control to be justified by precedent."""

    precedent: str = ""


@dataclass
class OutcomeSpec:
    """An outcome variable (§6.2), one per hypothesis.

    files:  one file, or several whose values are ADDED per country-year - the
            FDI ratio is inflows plus outflows (File 1 §3.1). A country-year
            missing any part is missing, never a partial sum.
    exclude_mean_above:
            leave a market out of THIS outcome's regressions when its mean
            value over the window is above this number. The market stays in
            the sample, the TSI pool and every other regression. Used for the
            FDI ratio, where financial centres record pass-through capital of
            several times their GDP (§6.2). None = no rule.
    """

    files: Tuple[str, ...]
    hypothesis: str
    label: str
    unit: str
    source: str = ""
    folder: str = "worldbank"
    exclude_mean_above: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.files, str):
            self.files = (self.files,)
        if not self.files:
            raise ValueError("an outcome needs at least one file")
        if self.hypothesis not in ("H1", "H2"):
            raise ValueError(f"hypothesis must be 'H1' or 'H2', "
                             f"got {self.hypothesis!r}")
