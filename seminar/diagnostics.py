"""§7.4: the diagnostic tests, and the standard-error remedy they select.

Every test runs on regression 1 (pooled baseline) of one outcome:

    Hausman              fixed vs random effects       H0: RE is consistent
    Fisher-ADF           panel unit root, Maddala-Wu   H0: every market's series
                         (and Choi's Z)                    has a unit root
    Pesaran CD           cross-sectional dependence    H0: residuals independent
                                                           across markets
    Modified Wald        groupwise heteroskedasticity  H0: equal variance in
                         (Greene)                          every market
    Wooldridge           first-order serial correlation H0: no AR(1)
    VIF                  multicollinearity             raw and within (after
                                                       the fixed effects)

The remedy (§7.4): Driscoll-Kraay standard errors when the CD test rejects
independence - they are robust to cross-sectional dependence as well as to
heteroskedasticity and serial correlation - and country-clustered errors
otherwise, which already cover heteroskedasticity and within-market serial
correlation.
"""

from __future__ import annotations

import warnings
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

from .regression import SE_CLUSTER, SE_DRISCOLL_KRAAY, Fit

ALPHA = 0.05


def _row(test, statistic, df, p, null, decision, note="") -> Dict:
    return {"test": test, "statistic": statistic, "df": df, "p_value": p,
            "null_hypothesis": null, "decision": decision, "note": note}


def _decision(p: float, reject: str, keep: str) -> str:
    if p is None or not np.isfinite(p):
        return "not computed"
    return reject if p < ALPHA else keep


# --------------------------------------------------------------------------
# Hausman: fixed vs random effects
# --------------------------------------------------------------------------

def hausman(data: pd.DataFrame, outcome: str, terms: Sequence[str]) -> Dict:
    """FE (within) vs RE (Swamy-Arora, unbalanced) on `terms`, with year
    dummies as regressors in both.

    Both covariance matrices use the disturbance variance of the efficient
    (RE) estimator - Stata's `hausman, sigmamore`. With each model's own
    variance, V_FE - V_RE is often not positive definite exactly when RE is
    inconsistent, and the statistic turns negative."""
    data = data.sort_values(["iso3", "year"]).reset_index(drop=True)
    years = pd.get_dummies(data["year"], prefix="yr", drop_first=True,
                           dtype=float)
    X = pd.concat([data[list(terms)].astype(float), years], axis=1)
    y = data[outcome].astype(float)
    ids = data["iso3"]
    k = X.shape[1]
    n, n_groups = len(data), ids.nunique()
    t_i = ids.map(ids.value_counts()).to_numpy(dtype=float)

    # within
    Xw = X - X.groupby(ids).transform("mean")
    yw = y - y.groupby(ids).transform("mean")
    keep = Xw.columns[Xw.abs().max() > 1e-12]
    Xw, X = Xw[keep], X[keep]
    k = X.shape[1]
    b_fe, *_ = np.linalg.lstsq(Xw.to_numpy(), yw.to_numpy(), rcond=None)
    e_fe = yw.to_numpy() - Xw.to_numpy() @ b_fe
    s2_e = e_fe @ e_fe / (n - n_groups - k)

    # between
    Xb = sm.add_constant(X.groupby(ids).mean())
    yb = y.groupby(ids).mean()
    b_b, *_ = np.linalg.lstsq(Xb.to_numpy(), yb.to_numpy(), rcond=None)
    e_b = yb.to_numpy() - Xb.to_numpy() @ b_b
    s2_b = e_b @ e_b / max(n_groups - k - 1, 1)
    t_bar = n_groups / np.sum(1.0 / ids.value_counts().to_numpy())
    s2_u = max(s2_b - s2_e / t_bar, 0.0)

    # quasi-demeaned GLS
    theta = 1 - np.sqrt(s2_e / (t_i * s2_u + s2_e))
    X_re = X.to_numpy() - theta[:, None] * X.groupby(ids).transform(
        "mean").to_numpy()
    X_re = np.column_stack([1 - theta, X_re])
    y_re = y.to_numpy() - theta * y.groupby(ids).transform("mean").to_numpy()
    b_re, *_ = np.linalg.lstsq(X_re, y_re, rcond=None)
    e_re = y_re - X_re @ b_re
    s2_re = e_re @ e_re / (n - k - 1)
    V_re = s2_re * np.linalg.pinv(X_re.T @ X_re)
    V_fe = s2_re * np.linalg.pinv(Xw.to_numpy().T @ Xw.to_numpy())

    compared = [i for i, c in enumerate(keep) if c in terms]
    d = b_fe[compared] - b_re[1:][compared]
    V_d = V_fe[np.ix_(compared, compared)] - V_re[1:, 1:][np.ix_(compared,
                                                                  compared)]
    statistic = float(d @ np.linalg.pinv(V_d) @ d)
    df = int(np.linalg.matrix_rank(V_d))
    note = ""
    if statistic < 0:
        note = ("V_FE - V_RE is not positive definite; the statistic is not "
                "reliable (common in small or unbalanced panels)")
    p = float(stats.chi2.sf(max(statistic, 0.0), df)) if df else np.nan
    return _row("Hausman (FE vs RE)", statistic, df, p,
                "random effects is consistent",
                _decision(p, "fixed effects (RE inconsistent)",
                          "RE not rejected; FE kept (consistent either way)"),
                note)


# --------------------------------------------------------------------------
# panel unit root: Maddala-Wu Fisher-ADF, with Choi's inverse-normal Z
# --------------------------------------------------------------------------

def fisher_adf(data: pd.DataFrame, column: str, lags: int = 1) -> Dict:
    p_values = []
    for _, series in data.sort_values("year").groupby("iso3")[column]:
        series = series.dropna().astype(float)
        if len(series) < 10 or series.std(ddof=0) == 0:
            continue
        try:
            p = adfuller(series.to_numpy(), maxlag=lags, regression="c",
                         autolag=None)[1]
        except Exception:                                # noqa: BLE001
            continue
        p_values.append(min(max(p, 1e-10), 1 - 1e-10))
    n = len(p_values)
    if n == 0:
        return _row(f"Fisher-ADF unit root: {column}", np.nan, np.nan, np.nan,
                    "all series have a unit root", "not computed")
    statistic = float(-2 * np.sum(np.log(p_values)))
    p = float(stats.chi2.sf(statistic, 2 * n))
    choi_z = float(np.sum(stats.norm.ppf(p_values)) / np.sqrt(n))
    return _row(f"Fisher-ADF unit root: {column}", statistic, 2 * n, p,
                "all market series have a unit root",
                _decision(p, "stationary in at least some markets",
                          "unit root not rejected"),
                f"{n} markets with a usable series; ADF({lags}) with "
                f"constant; Choi Z = {choi_z:.2f} "
                f"(p = {stats.norm.cdf(choi_z):.3f})")


# --------------------------------------------------------------------------
# residual-based tests
# --------------------------------------------------------------------------

def _residual_frame(fit: Fit, data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["iso3", "year"]).reset_index(drop=True)
    return pd.DataFrame({"iso3": data["iso3"], "year": data["year"],
                         "e": np.asarray(fit.result.resid)})


def pesaran_cd(residuals: pd.DataFrame, min_overlap: int = 3) -> Dict:
    wide = residuals.pivot(index="year", columns="iso3", values="e")
    values = wide.to_numpy()
    present = ~np.isnan(values)
    n = values.shape[1]
    total, pairs, abs_rho = 0.0, 0, []
    for i in range(n - 1):
        for j in range(i + 1, n):
            both = present[:, i] & present[:, j]
            t_ij = int(both.sum())
            if t_ij < min_overlap:
                continue
            a, b = values[both, i], values[both, j]
            if a.std() == 0 or b.std() == 0:
                continue
            rho = float(np.corrcoef(a, b)[0, 1])
            total += np.sqrt(t_ij) * rho
            abs_rho.append(abs(rho))
            pairs += 1
    if pairs == 0:
        return _row("Pesaran CD", np.nan, np.nan, np.nan,
                    "no cross-sectional dependence", "not computed")
    statistic = float(np.sqrt(2.0 / (n * (n - 1))) * total)
    p = float(2 * stats.norm.sf(abs(statistic)))
    return _row("Pesaran CD", statistic, np.nan, p,
                "no cross-sectional dependence",
                _decision(p, "cross-sectional dependence",
                          "independence not rejected"),
                f"{pairs} market pairs; mean |rho| = {np.mean(abs_rho):.3f}")


def modified_wald(residuals: pd.DataFrame) -> Dict:
    e2 = residuals["e"] ** 2
    overall = float(e2.mean())
    statistic, units = 0.0, 0
    for _, chunk in e2.groupby(residuals["iso3"]):
        t_i = len(chunk)
        if t_i < 2:
            continue
        s2_i = float(chunk.mean())
        v_i = float(((chunk - s2_i) ** 2).sum() / (t_i * (t_i - 1)))
        if v_i <= 0:
            continue
        statistic += (s2_i - overall) ** 2 / v_i
        units += 1
    p = float(stats.chi2.sf(statistic, units)) if units else np.nan
    return _row("Modified Wald (groupwise heteroskedasticity)", statistic,
                units, p, "equal error variance in every market",
                _decision(p, "heteroskedastic", "homoskedasticity not "
                                                 "rejected"))


def wooldridge_ar1(data: pd.DataFrame, outcome: str,
                   terms: Sequence[str]) -> Dict:
    """Regress first differences (with year dummies), then test that the
    differenced residuals have autocorrelation -0.5, clustered by market."""
    data = data.sort_values(["iso3", "year"]).reset_index(drop=True)
    columns = [outcome] + list(terms)
    consecutive = data.groupby("iso3")["year"].diff() == 1
    diffs = data.groupby("iso3")[columns].diff()[consecutive]
    diffs["iso3"] = data.loc[consecutive, "iso3"]
    diffs["year"] = data.loc[consecutive, "year"]
    diffs = diffs.dropna().reset_index(drop=True)
    usable = [t for t in terms if diffs[t].std(ddof=0) > 0]
    formula = f"{outcome} ~ " + " + ".join(usable + ["C(year)"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = smf.ols(formula, data=diffs).fit()
    frame = pd.DataFrame({"iso3": diffs["iso3"], "year": diffs["year"],
                          "u": np.asarray(first.resid)})
    frame["u_lag"] = frame.groupby("iso3")["u"].shift(1)
    frame["gap"] = frame.groupby("iso3")["year"].diff()
    frame = frame[(frame["gap"] == 1)].dropna().reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        second = smf.ols("u ~ u_lag - 1", data=frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": pd.factorize(frame["iso3"])[0]})
    coef, se = float(second.params["u_lag"]), float(second.bse["u_lag"])
    statistic = ((coef + 0.5) / se) ** 2
    groups = frame["iso3"].nunique()
    p = float(stats.f.sf(statistic, 1, groups - 1))
    return _row("Wooldridge AR(1)", statistic, f"(1, {groups - 1})", p,
                "no first-order serial correlation",
                _decision(p, "serial correlation", "no serial correlation"),
                f"coefficient on lagged differenced residual {coef:.3f} "
                f"(-0.5 under H0)")


# --------------------------------------------------------------------------
# multicollinearity
# --------------------------------------------------------------------------

def _within(data: pd.DataFrame, column: str) -> np.ndarray:
    """The column after removing country and year fixed effects exactly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols(f"{column} ~ C(iso3) + C(year)", data=data).fit()
    return np.asarray(fit.resid)


def vif_table(data: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    data = data.reset_index(drop=True)
    terms = [t for t in terms if data[t].std(ddof=0) > 0]
    rows = []
    raw = sm.add_constant(data[terms].astype(float)).to_numpy()
    within = np.column_stack([_within(data, t) for t in terms])
    for index, term in enumerate(terms):
        try:
            v_raw = variance_inflation_factor(raw, index + 1)
        except Exception:                                # noqa: BLE001
            v_raw = np.nan
        try:
            v_within = variance_inflation_factor(within, index)
        except Exception:                                # noqa: BLE001
            v_within = np.nan
        rows.append({"term": term, "vif_raw": v_raw, "vif_within": v_within})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the whole battery for one outcome
# --------------------------------------------------------------------------

def run_diagnostics(data: pd.DataFrame, outcome: str, terms: Sequence[str],
                    fit: Fit, se_setting: str
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """All §7.4 tests on regression 1 of one outcome.

    Returns (tests, vif, se_type). se_setting "auto" applies the remedy rule;
    "cluster" or "driscoll_kraay" force one type and record that.
    """
    residuals = _residual_frame(fit, data)
    rows = [
        hausman(data, outcome, terms),
        fisher_adf(data, outcome),
        fisher_adf(data, terms[0]),                    # the treatment, TSI
        pesaran_cd(residuals),
        modified_wald(residuals),
        wooldridge_ar1(data, outcome, terms),
    ]
    vif = vif_table(data, terms)
    rows.append(_row("VIF (largest, within)", float(vif["vif_within"].max()),
                     np.nan, np.nan, "", "above 10 signals collinearity"
                     if vif["vif_within"].max() > 10 else "no serious "
                     "collinearity", "full table in t8_3_vif.csv"))
    tests = pd.DataFrame(rows)
    tests.insert(0, "outcome", outcome)

    if se_setting == "auto":
        cd = tests.loc[tests["test"] == "Pesaran CD", "p_value"].iloc[0]
        se_type = (SE_DRISCOLL_KRAAY if np.isfinite(cd) and cd < ALPHA
                   else SE_CLUSTER)
        reason = ("Pesaran CD rejects independence -> Driscoll-Kraay"
                  if se_type == SE_DRISCOLL_KRAAY else
                  "no cross-sectional dependence detected -> clustered by "
                  "country (robust to heteroskedasticity and serial "
                  "correlation within a market)")
    else:
        se_type = se_setting
        reason = f"set in config.py (SE_TYPE = '{se_setting}')"
    tests = pd.concat([tests, pd.DataFrame([_row(
        "Standard errors adopted", np.nan, np.nan, np.nan, "",
        {SE_CLUSTER: "clustered by country",
         SE_DRISCOLL_KRAAY: "Driscoll-Kraay"}[se_type], reason)])
        .assign(outcome=outcome)], ignore_index=True)
    return tests, vif, se_type
