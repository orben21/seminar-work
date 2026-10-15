"""§7: the regressions.

The analysis is four regressions (§7.1), each run once per outcome (GDP
growth for H1, the FDI ratio for H2) - eight in all:

    1  pooled baseline          Y = b1 TSI(t) + b2 TSI(t-1) + g'X + mu_i + la_t + e
    2  developed markets only   the same equation on the developed rows
    3  developing markets only  the same equation on the developing rows
    4  pooled with interaction  1 + b3 TSI(t) x DEV + b4 TSI(t-1) x DEV

In regression 4 developing markets are the reference group: b1 is the effect
there, b1 + b3 the effect in developed markets, and b3 - the difference - is
the formal test of H1 and H2 (§7.3). DEV never enters alone: it is fixed per
market, so the country fixed effects absorb it.

Around them, as §8 asks: the specification build-up (§8.4/§8.5), the
component decomposition (§8.7) and the hypothesis-test summary (§8.8). Every
regression of one outcome is estimated on the same rows and with the same
standard-error type (§7.4 decides which).

Fixed effects are dummy variables (C(iso3), C(year)). Coefficients on the TSI
are per ONE STANDARD DEVIATION of terrorism severity in this sample (§5.6).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .panel import (COMPONENT_TERMS, INTERACTION, INTERACTION_LAG, TREATMENT,
                    TREATMENT_LAG, Panel, complete_rows)
from .tsi import COMPONENTS

SE_CLUSTER = "cluster"
SE_DRISCOLL_KRAAY = "driscoll_kraay"
SE_LABELS = {SE_CLUSTER: "clustered by country",
             SE_DRISCOLL_KRAAY: "Driscoll-Kraay"}

CORE_TITLES = {
    "1_pooled": "Regression 1 - pooled baseline",
    "2_developed": "Regression 2 - developed markets only",
    "3_developing": "Regression 3 - developing markets only",
    "4_interaction": "Regression 4 - pooled with interaction",
}
# The plan's coefficient labels (§7.2-§7.3).
BETA = {TREATMENT: "b1", TREATMENT_LAG: "b2", INTERACTION: "b3",
        INTERACTION_LAG: "b4"}


@dataclass
class Fit:
    key: str
    title: str
    outcome: str
    sample: str
    terms: List[str]
    formula: str
    se_type: str
    n_obs: int
    n_markets: int
    n_years: int
    r2: float
    r2_within: float
    coefficients: pd.DataFrame
    notes: List[str] = field(default_factory=list)
    result: object = field(repr=False, default=None)

    def get(self, term: str) -> Optional[pd.Series]:
        rows = self.coefficients[self.coefficients["term"] == term]
        return None if rows.empty else rows.iloc[0]

    def combination(self, terms: Sequence[str]) -> Dict[str, float]:
        """Estimate, SE, p and 95% CI of a sum of coefficients."""
        present = [t for t in terms if t in self.result.params.index]
        if len(present) != len(terms):
            return {}
        test = self.result.t_test(" + ".join(present) + " = 0")
        low, high = np.squeeze(test.conf_int())
        return {"coef": float(np.squeeze(test.effect)),
                "se": float(np.squeeze(test.sd)),
                "p": float(np.squeeze(test.pvalue)),
                "ci_low": float(low), "ci_high": float(high)}

    def joint_test(self, terms: Sequence[str]) -> Dict[str, float]:
        present = [t for t in terms if t in self.result.params.index]
        if len(present) != len(terms):
            return {}
        test = self.result.wald_test(", ".join(f"{t} = 0" for t in present),
                                     use_f=True, scalar=True)
        return {"F": float(np.squeeze(test.statistic)),
                "p": float(np.squeeze(test.pvalue)),
                "df_num": float(test.df_num), "df_denom": float(test.df_denom)}


def stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def dk_maxlags(n_periods: int) -> int:
    """Newey-West rule floor(4 (T/100)^(2/9)), the usual Driscoll-Kraay lag."""
    return max(1, int(math.floor(4 * (n_periods / 100) ** (2 / 9))))


def _formula(outcome: str, terms: Sequence[str], country_fe: bool,
             year_fe: bool) -> str:
    parts = list(terms)
    if country_fe:
        parts.append("C(iso3)")
    if year_fe:
        parts.append("C(year)")
    return f"{outcome} ~ " + (" + ".join(parts) if parts else "1")


def _usable_terms(data: pd.DataFrame, terms: Sequence[str],
                  notes: List[str]) -> List[str]:
    """Drop regressors that cannot be estimated on these rows: constant, or
    with no within-market variation (the country dummies absorb them)."""
    kept = []
    for term in terms:
        column = data[term].astype(float)
        if column.std(ddof=0) == 0:
            notes.append(f"{term} is constant in this sample and was dropped")
            continue
        within = column.groupby(data["iso3"]).transform(lambda s: s - s.mean())
        if within.abs().max() < 1e-12:
            notes.append(f"{term} never varies within a market in this sample "
                         f"and was dropped")
            continue
        kept.append(term)
    return kept


def fit_fe(data: pd.DataFrame, outcome: str, terms: Sequence[str],
           se_type: str = SE_CLUSTER, country_fe: bool = True,
           year_fe: bool = True, key: str = "", title: str = "",
           sample: str = "all") -> Fit:
    """OLS with country and/or year dummies and robust standard errors.

    cluster          clustered by country; t-tests with G - 1 df
    driscoll_kraay   robust to heteroskedasticity, serial correlation and
                     cross-sectional dependence (Bartlett kernel, dk_maxlags
                     lags); t-tests with T - 1 df, since the number of time
                     periods is this estimator's effective sample size
    """
    data = data.sort_values(["iso3", "year"]).reset_index(drop=True)
    notes: List[str] = []
    terms = _usable_terms(data, terms, notes)
    formula = _formula(outcome, terms, country_fe, year_fe)
    n_years = int(data["year"].nunique())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=data)
        if se_type == SE_DRISCOLL_KRAAY:
            codes = data["year"].map({y: i for i, y in enumerate(
                sorted(data["year"].unique()))}).to_numpy()
            result = model.fit(cov_type="hac-groupsum",
                               cov_kwds={"time": codes,
                                         "maxlags": dk_maxlags(n_years),
                                         "use_correction": "hac"},
                               use_t=True)
            # set on the underlying results: on the wrapper statsmodels ignores it
            result._results.df_resid_inference = n_years - 1
        else:
            groups = pd.factorize(data["iso3"])[0]
            result = model.fit(cov_type="cluster",
                               cov_kwds={"groups": groups}, use_t=True)

        fe_only = _formula(outcome, [], country_fe, year_fe)
        rss_fe = smf.ols(fe_only, data=data).fit().ssr

    conf = result.conf_int()
    table = pd.DataFrame({
        "term": result.params.index, "coef": result.params.values,
        "se": result.bse.values, "t": result.tvalues.values,
        "p": result.pvalues.values, "ci_low": conf.iloc[:, 0].values,
        "ci_high": conf.iloc[:, 1].values,
    })
    table = table[~table["term"].str.startswith(("C(iso3)", "C(year)"))]
    table["stars"] = table["p"].map(stars)
    table["beta"] = table["term"].map(BETA).fillna("")

    n_markets = int(data["iso3"].nunique())
    if se_type == SE_CLUSTER and n_markets < 20:
        notes.append(f"only {n_markets} clusters: clustered standard errors "
                     f"may be too small")
    return Fit(key=key, title=title, outcome=outcome, sample=sample,
               terms=list(terms), formula=formula, se_type=se_type,
               n_obs=int(result.nobs), n_markets=n_markets, n_years=n_years,
               r2=float(result.rsquared),
               r2_within=float(1 - result.ssr / rss_fe) if rss_fe > 0
               else np.nan,
               coefficients=table.reset_index(drop=True), notes=notes,
               result=result)


# --------------------------------------------------------------------------
# the four regressions of §7.1
# --------------------------------------------------------------------------

def estimation_sample(panel: Panel, outcome: str,
                      extra: Sequence[str] = ()) -> pd.DataFrame:
    """Rows with the outcome, TSI(t), TSI(t-1), the group and every control.
    The first window year drops out: its lag lies before the window."""
    columns = ([outcome, TREATMENT, TREATMENT_LAG, "developed"]
               + panel.control_names + list(extra))
    return complete_rows(panel.data, columns)


def core_regressions(panel: Panel, outcome: str, se_type: str,
                     exclude: Sequence[str] = ()) -> Dict[str, Fit]:
    """Regressions 1-4 for one outcome, on one common set of rows."""
    base = estimation_sample(panel, outcome)
    if exclude:
        base = base[~base["iso3"].isin(exclude)].reset_index(drop=True)
    controls = panel.control_names
    fits = {}
    fits["1_pooled"] = fit_fe(base, outcome,
                              [TREATMENT, TREATMENT_LAG] + controls, se_type,
                              key="1_pooled", title=CORE_TITLES["1_pooled"])
    for key, flag, sample in (("2_developed", 1.0, "developed"),
                              ("3_developing", 0.0, "developing")):
        rows = base[base["developed"] == flag]
        if rows["iso3"].nunique() < 2:
            continue
        fits[key] = fit_fe(rows, outcome, [TREATMENT, TREATMENT_LAG] + controls,
                           se_type, key=key, title=CORE_TITLES[key],
                           sample=sample)
    if base["developed"].nunique() == 2:
        fits["4_interaction"] = fit_fe(
            base, outcome,
            [TREATMENT, TREATMENT_LAG, INTERACTION, INTERACTION_LAG] + controls,
            se_type, key="4_interaction", title=CORE_TITLES["4_interaction"])
    return fits


def marginal_effects(fit4: Fit) -> pd.DataFrame:
    """§8.6 / Figure 22: the effect inside each group, and the difference."""
    rows = []
    for horizon, own, inter in (("t", TREATMENT, INTERACTION),
                                ("t-1", TREATMENT_LAG, INTERACTION_LAG)):
        for group, terms in (("developing", [own]),
                             ("developed", [own, inter]),
                             ("difference (developed - developing)", [inter])):
            estimate = fit4.combination(terms)
            if estimate:
                rows.append({"outcome": fit4.outcome, "horizon": horizon,
                             "group": group,
                             "combination": " + ".join(BETA[t] for t in terms),
                             **estimate, "stars": stars(estimate["p"])})
    return pd.DataFrame(rows)


def buildup(panel: Panel, outcome: str, se_type: str) -> List[Fit]:
    """§8.4 / §8.5: TSI only, + controls, + year FE, + lag - on the rows of
    regression 1, so the columns differ in specification and nothing else.
    Country fixed effects are in every column (the empirical strategy of §7.1).
    """
    base = estimation_sample(panel, outcome)
    controls = panel.control_names
    columns = [
        ("(1) TSI only", [TREATMENT], False),
        ("(2) + controls", [TREATMENT] + controls, False),
        ("(3) + year FE", [TREATMENT] + controls, True),
        ("(4) + lag", [TREATMENT, TREATMENT_LAG] + controls, True),
    ]
    return [fit_fe(base, outcome, terms, se_type, year_fe=year_fe,
                   key=f"buildup_{index}", title=title)
            for index, (title, terms, year_fe) in enumerate(columns, start=1)]


def component_regressions(panel: Panel, outcome: str, se_type: str,
                          mode: str = "separate") -> List[Fit]:
    """§8.7: the four components in place of the composite.

    separate  one regression per component: z(ln(1+x)) at t and t-1 replace
              TSI(t) and TSI(t-1). Each coefficient is on the same one-SD
              scale as the component enters the TSI.
    joint     all four components (and lags) in one regression. They are
              strongly correlated, so read the VIFs before the coefficients.
    """
    lags = [f"{term}_lag1" for term in COMPONENT_TERMS]
    base = estimation_sample(panel, outcome, extra=COMPONENT_TERMS + lags)
    controls = panel.control_names
    samples = [("all", base), ("developed", base[base["developed"] == 1.0]),
               ("developing", base[base["developed"] == 0.0])]
    fits = []
    for sample, rows in samples:
        if rows["iso3"].nunique() < 2:
            continue
        if mode == "joint":
            fits.append(fit_fe(rows, outcome, COMPONENT_TERMS + lags + controls,
                               se_type, key=f"components_{sample}",
                               title="all four components", sample=sample))
            continue
        for component, term in zip(COMPONENTS, COMPONENT_TERMS):
            fits.append(fit_fe(rows, outcome, [term, f"{term}_lag1"]
                               + controls, se_type,
                               key=f"component_{component}_{sample}",
                               title=f"{component} in place of the TSI",
                               sample=sample))
    return fits


def component_table(fits: List[Fit]) -> pd.DataFrame:
    rows = []
    for fit in fits:
        for component, term in zip(COMPONENTS, COMPONENT_TERMS):
            for horizon, name in (("t", term), ("t-1", f"{term}_lag1")):
                row = fit.get(name)
                if row is None:
                    continue
                rows.append({"outcome": fit.outcome, "sample": fit.sample,
                             "component": component, "horizon": horizon,
                             "coef": row["coef"], "se": row["se"],
                             "p": row["p"], "ci_low": row["ci_low"],
                             "ci_high": row["ci_high"], "stars": row["stars"],
                             "n_obs": fit.n_obs, "specification": fit.title})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# tables for §8
# --------------------------------------------------------------------------

def coefficient_columns(fits: List[Fit], terms: Sequence[str]) -> pd.DataFrame:
    """A regression table: one column per fit, coef (se) stars per term,
    then N, markets, R2 and within R2."""
    table = {}
    for fit in fits:
        cells = {}
        for term in terms:
            row = fit.get(term)
            cells[term] = ("" if row is None else
                           f"{row['coef']:.3f}{row['stars']} ({row['se']:.3f})")
        cells["observations"] = f"{fit.n_obs:,}"
        cells["markets"] = str(fit.n_markets)
        cells["years"] = str(fit.n_years)
        cells["R2"] = f"{fit.r2:.3f}"
        cells["within R2"] = f"{fit.r2_within:.3f}"
        cells["country FE"] = "yes" if "C(iso3)" in fit.formula else "no"
        cells["year FE"] = "yes" if "C(year)" in fit.formula else "no"
        cells["SE"] = SE_LABELS[fit.se_type]
        table[fit.title] = cells
    frame = pd.DataFrame(table)
    frame.index.name = "term"
    return frame.reset_index()


def core_terms(panel: Panel) -> List[str]:
    return [TREATMENT, TREATMENT_LAG, INTERACTION, INTERACTION_LAG] \
        + panel.control_names


def hypothesis_tests(core: Dict[str, Dict[str, Fit]],
                     hypotheses: Dict[str, str],
                     alpha: float = 0.05) -> pd.DataFrame:
    """§8.8: hypothesis, specification, coefficient, SE, p, decision."""
    rows = []
    for outcome, fits in core.items():
        fit = fits.get("4_interaction")
        if fit is None:
            continue
        hypothesis = hypotheses[outcome]
        for term, label in ((INTERACTION, "b3: TSI(t) x Developed"),
                            (INTERACTION_LAG, "b4: TSI(t-1) x Developed")):
            row = fit.get(term)
            if row is None:
                continue
            reject = row["p"] < alpha
            rows.append({
                "hypothesis": hypothesis, "outcome": outcome,
                "specification": fit.title, "quantity": label,
                "coef": row["coef"], "se": row["se"], "p": row["p"],
                "decision": (f"reject H0 ({label.split(':')[0]} = 0) at "
                             f"{alpha:.0%}" if reject else
                             f"do not reject H0 ({label.split(':')[0]} = 0)"),
            })
        joint = fit.joint_test([INTERACTION, INTERACTION_LAG])
        if joint:
            rows.append({
                "hypothesis": hypothesis, "outcome": outcome,
                "specification": fit.title,
                "quantity": "b3 = b4 = 0 (differ at any horizon)",
                "coef": np.nan, "se": np.nan, "p": joint["p"],
                "F": joint["F"],
                "decision": ("reject H0 (b3 = b4 = 0)" if joint["p"] < alpha
                             else "do not reject H0 (b3 = b4 = 0)"),
            })
        base = fit.get(TREATMENT)
        if base is not None:
            rows.append({
                "hypothesis": hypothesis, "outcome": outcome,
                "specification": fit.title,
                "quantity": "b1: TSI(t) in developing markets (reference)",
                "coef": base["coef"], "se": base["se"], "p": base["p"],
                "decision": "reported alongside (Part 0.4)",
            })
    return pd.DataFrame(rows)


def all_coefficients(fits: List[Fit]) -> pd.DataFrame:
    """Appendix D: every non-dummy coefficient of every regression."""
    frames = []
    for fit in fits:
        table = fit.coefficients.copy()
        table.insert(0, "sample", fit.sample)
        table.insert(0, "specification", fit.title)
        table.insert(0, "outcome", fit.outcome)
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def model_table(fits: List[Fit]) -> pd.DataFrame:
    return pd.DataFrame([{
        "outcome": fit.outcome, "specification": fit.title,
        "sample": fit.sample, "n_obs": fit.n_obs, "markets": fit.n_markets,
        "years": fit.n_years, "r2": fit.r2, "r2_within": fit.r2_within,
        "se": SE_LABELS[fit.se_type], "formula": fit.formula,
        "notes": "; ".join(fit.notes),
    } for fit in fits])
