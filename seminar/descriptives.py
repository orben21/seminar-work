"""§6.6: descriptive statistics and the correlation matrix.

The difference-in-means test is a regression of the variable on the developed
dummy with standard errors clustered by market: market-years of one country
are not independent draws, so a plain two-sample t-test would overstate the
evidence.
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

STATS = ["N", "mean", "sd", "min", "p25", "median", "p75", "max"]


def _summary(values: pd.Series) -> Dict[str, float]:
    values = values.dropna().astype(float)
    if values.empty:
        return {name: np.nan for name in STATS}
    return {"N": len(values), "mean": values.mean(), "sd": values.std(ddof=1),
            "min": values.min(), "p25": values.quantile(0.25),
            "median": values.median(), "p75": values.quantile(0.75),
            "max": values.max()}


def _difference_test(data: pd.DataFrame, column: str) -> Dict[str, float]:
    frame = data[["iso3", "developed", column]].dropna()
    if frame["developed"].nunique() < 2 or frame[column].std(ddof=0) == 0:
        return {"diff_developed_minus_developing": np.nan, "diff_p": np.nan}
    frame = frame.rename(columns={column: "value"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("value ~ developed", data=frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": pd.factorize(frame["iso3"])[0]}, use_t=True)
    return {"diff_developed_minus_developing": float(fit.params["developed"]),
            "diff_p": float(fit.pvalues["developed"])}


def describe_by_group(data: pd.DataFrame, variables: List[str],
                      labels: Dict[str, str]) -> pd.DataFrame:
    """One row per variable: the eight statistics for the developed and the
    developing group, then the difference in means and its p-value."""
    rows = []
    for column in variables:
        row = {"variable": column, "label": labels.get(column, column)}
        for group, flag in (("developed", 1.0), ("developing", 0.0)):
            for stat, value in _summary(
                    data.loc[data["developed"] == flag, column]).items():
                row[f"{group}_{stat}"] = value
        row.update(_difference_test(data, column))
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_matrix(data: pd.DataFrame, variables: List[str]
                       ) -> pd.DataFrame:
    """Pearson correlations, pairwise complete."""
    return data[variables].astype(float).corr()


def extreme_values(data: pd.DataFrame, column: str,
                   top: int = 15) -> pd.DataFrame:
    """The market-years furthest from zero - where a few observations could
    carry an estimate (financial centres can post FDI flows of several times
    their GDP)."""
    frame = data[["iso3", "country", "year", "group", column]].dropna()
    order = frame[column].abs().sort_values(ascending=False).index
    frame = frame.loc[order].head(top).rename(columns={column: "value"})
    return frame.assign(variable=column).reset_index(drop=True)
