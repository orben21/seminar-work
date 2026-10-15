"""File 3, Part B: the Terrorism Severity Index (TSI).

For every market i and year t, from the four aggregates of terror.py:

    Step 1  NA, NK, NI, PD                       (terror.aggregate_market_years)
    Step 2  x~ = ln(1 + x)                       compresses the skew, keeps
                                                 zero market-years at zero
    Step 3  z(x~) = (x~ - mean) / sd             mean and sd over the POOLED
                                                 panel: every market x every
                                                 year of the frozen sample,
                                                 both groups together
    Step 4  TSI = 1/4 [z(NA) + z(NK) + z(NI) + z(PD)]

Rules the plan attaches to the construction (§5.6, §5.7):

  * The sample must be frozen before this runs - adding or dropping one market
    moves every TSI value. build_tsi() is only ever given the admitted sample.
  * Market-years with no qualifying event stay in the pool (they are the most
    negative TSI, not zero). Only genuinely missing component values - NK or NI
    of a market-year whose every event has a blank count - are left out, and
    only from that component's mean and sd. Such a market-year's TSI is the
    mean of the components it has; how many there are is reported.
  * The standard deviation is the sample sd (n - 1).
  * No population scaling (§5.7): the index uses raw aggregates.

Coefficients on the TSI therefore read in standard-deviation units of this
sample: "a one-SD rise in terrorism severity is associated with ...".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

COMPONENTS = ["NA", "NK", "NI", "PD"]
COMPONENT_LABELS = {
    "NA": "qualifying attacks",
    "NK": "fatalities",
    "NI": "injured",
    "PD": "property damage score",
}


@dataclass
class TSIResult:
    table: pd.DataFrame      # iso3, year, components, ln_*, z_*, TSI
    moments: pd.DataFrame    # Appendix D: the four means and sds
    skewness: pd.DataFrame   # Appendix D: shape before and after the log


def build_tsi(country_year: pd.DataFrame) -> TSIResult:
    """Steps 2-4 on the frozen sample's market-year table."""
    table = country_year.copy()
    moment_rows = []
    for component in COMPONENTS:
        logged = np.log1p(table[component].astype(float))
        mean, sd = logged.mean(), logged.std(ddof=1)
        table[f"ln_{component}"] = logged
        table[f"z_{component}"] = (logged - mean) / sd if sd > 0 else np.nan
        moment_rows.append({
            "component": component,
            "label": COMPONENT_LABELS[component],
            "n_market_years": int(logged.notna().sum()),
            "n_missing": int(logged.isna().sum()),
            "mean_ln1p": mean,
            "sd_ln1p": sd,
            "raw_mean": table[component].mean(),
            "raw_sd": table[component].std(ddof=1),
            "raw_max": table[component].max(),
            "share_zero": float((table[component] == 0).mean()),
        })

    z_columns = [f"z_{c}" for c in COMPONENTS]
    table["n_components"] = table[z_columns].notna().sum(axis=1)
    table["TSI"] = table[z_columns].mean(axis=1, skipna=True)
    table.loc[table["n_components"] == 0, "TSI"] = np.nan

    return TSIResult(table=table,
                     moments=pd.DataFrame(moment_rows),
                     skewness=skewness_table(table))


def _shape(values: pd.Series) -> dict:
    """The §5.6 Step 2 statistics for one column of the pool."""
    values = values.dropna().astype(float)
    if len(values) < 3 or values.std(ddof=1) == 0:
        return {"skewness": np.nan, "max_sd_above_mean": np.nan,
                "share_below_mean": np.nan, "top_obs_share_of_ssd": np.nan}
    deviation = values - values.mean()
    squared = deviation ** 2
    return {
        "skewness": float(values.skew()),
        "max_sd_above_mean": float(deviation.max() / values.std(ddof=1)),
        "share_below_mean": float((deviation < 0).mean()),
        "top_obs_share_of_ssd": float(squared.max() / squared.sum()),
    }


def skewness_table(table: pd.DataFrame) -> pd.DataFrame:
    """Skewness and concentration of each component, raw vs ln(1+x) vs z.

    Standardisation is linear, so the z column has exactly the skewness of the
    log column - shown so the reader can see that Step 3 cannot do Step 2's job.
    """
    rows = []
    for component in COMPONENTS:
        for stage, column in (("raw", component),
                              ("ln(1+x)", f"ln_{component}"),
                              ("z-score", f"z_{component}")):
            if column in table.columns:
                rows.append({"component": component, "stage": stage,
                             **_shape(table[column])})
    return pd.DataFrame(rows)


def skewness_by_market(table: pd.DataFrame, component: str = "NK"
                       ) -> pd.DataFrame:
    """Figure 13A: skewness of each market's own raw annual series."""
    rows = []
    for iso3, series in table.groupby("iso3")[component]:
        series = series.dropna()
        skew = (float(series.skew()) if len(series) >= 3
                and series.std(ddof=1) > 0 else np.nan)
        rows.append({"iso3": iso3, "skewness": skew,
                     "total": float(series.sum()),
                     "years_nonzero": int((series > 0).sum())})
    return pd.DataFrame(rows)


def worked_example(table: pd.DataFrame, moments: pd.DataFrame,
                   events: Optional[pd.DataFrame] = None,
                   iso3: Optional[str] = None,
                   year: Optional[int] = None) -> pd.DataFrame:
    """Figure 13: one market-year carried from raw aggregates to the TSI.

    With no market-year given, picks the one with the median TSI among those
    whose four components are all positive - an ordinary active year rather
    than an extreme one.
    """
    if iso3 is None or year is None:
        active = table[(table[COMPONENTS] > 0).all(axis=1)
                       & table["TSI"].notna()]
        if active.empty:
            return pd.DataFrame()
        ordered = active.sort_values("TSI").reset_index(drop=True)
        pick = ordered.iloc[len(ordered) // 2]
        iso3, year = pick["iso3"], int(pick["year"])
    row = table[(table["iso3"] == iso3) & (table["year"] == year)]
    if row.empty:
        return pd.DataFrame()
    row = row.iloc[0]
    stats = moments.set_index("component")
    rows = []
    for component in COMPONENTS:
        rows.append({
            "iso3": iso3, "year": year, "component": component,
            "label": COMPONENT_LABELS[component],
            "step1_raw": row[component],
            "step2_ln1p": row[f"ln_{component}"],
            "pool_mean": stats.loc[component, "mean_ln1p"],
            "pool_sd": stats.loc[component, "sd_ln1p"],
            "step3_z": row[f"z_{component}"],
            "step4_weight": 0.25,
        })
    example = pd.DataFrame(rows)
    example["TSI"] = row["TSI"]
    if events is not None and not events.empty:
        ids = events[(events["iso3"] == iso3) & (events["iyear"] == year)
                     ]["eventid"].astype("int64").astype(str).tolist()
        example["eventids"] = " ".join(ids)
    return example


def add_lags(table: pd.DataFrame, columns: List[str],
             lag: int = 1) -> pd.DataFrame:
    """<column>_lag<k> per market. The first window year has no lag: the year
    before the window is outside the pool and outside K3's coding."""
    table = table.sort_values(["iso3", "year"]).copy()
    for column in columns:
        table[f"{column}_lag{lag}"] = table.groupby("iso3")[column].shift(lag)
    return table
