"""The regressions, run independently for every metric.

For each metric a ladder of nested models is estimated on exactly the same rows,
so the models are directly comparable:

    A  before terrorism   metric ~ controls + country FE + year FE
    B  after terrorism    metric ~ controls + terror + country FE + year FE
    C  heterogeneity      B + terror x developed
    D  long run           metric ~ controls + terror(t, t-1, ... t-k) + FE

The A -> B difference is the headline: what terrorism adds once everything else
is already accounted for.

Fixed effects are ordinary dummy variables (C(iso3), C(year)) rather than a
specialised panel package. Country dummies absorb everything permanent about a
country; year dummies absorb whatever hit the whole world that year. Standard
errors are clustered by country, because a country's shocks persist across years.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .panel import Panel

MODEL_NAMES = {
    "A": "before terrorism",
    "B": "after terrorism",
    "C": "developed vs developing",
    "D": "long run (distributed lag)",
}


@dataclass
class ModelFit:
    key: str
    name: str
    formula: str
    n_obs: int
    n_countries: int
    r2: float
    r2_within: float
    rss: float
    coefficients: pd.DataFrame
    result: object = field(repr=False, default=None)

    def coef(self, term: str) -> Optional[float]:
        row = self.coefficients[self.coefficients["term"] == term]
        return None if row.empty else float(row["coef"].iloc[0])


@dataclass
class MetricResult:
    """Everything computed for one metric."""

    metric: str
    label: str
    unit: str
    terror_measure: str
    n_obs: int
    n_countries: int
    year_min: int
    year_max: int
    models: Dict[str, ModelFit] = field(default_factory=dict)
    comparison: Dict[str, float] = field(default_factory=dict)
    control_shifts: pd.DataFrame = field(default_factory=pd.DataFrame)
    group_effects: pd.DataFrame = field(default_factory=pd.DataFrame)
    long_run: Dict[str, float] = field(default_factory=dict)
    lag_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    vif: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: Dict[str, float] = field(default_factory=dict)
    interpretation: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None


# --------------------------------------------------------------------------
# fitting helpers
# --------------------------------------------------------------------------

def _fit(formula: str, data: pd.DataFrame):
    """OLS with country-clustered standard errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=data)
        return model.fit(cov_type="cluster",
                         cov_kwds={"groups": data["iso3"]},
                         use_t=True)


def _two_way_demeaned(values: pd.Series, iso3: pd.Series,
                      year: pd.Series) -> pd.Series:
    """y minus country mean minus year mean plus grand mean."""
    grand = values.mean()
    by_country = values.groupby(iso3).transform("mean")
    by_year = values.groupby(year).transform("mean")
    return values - by_country - by_year + grand


def _tidy(result, drop_fixed_effects: bool = True) -> pd.DataFrame:
    """Coefficients as a flat table, with the hundreds of dummies left out."""
    table = pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "se": result.bse.values,
        "t": result.tvalues.values,
        "p": result.pvalues.values,
    })
    conf = result.conf_int()
    table["ci_low"] = conf.iloc[:, 0].values
    table["ci_high"] = conf.iloc[:, 1].values
    if drop_fixed_effects:
        is_dummy = table["term"].str.startswith(("C(iso3)", "C(year)"))
        table = table[~is_dummy]
    table["stars"] = table["p"].map(_stars)
    return table.reset_index(drop=True)


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _wald_p(result, terms: List[str]) -> Optional[float]:
    """Joint significance of `terms`, using the clustered covariance."""
    present = [t for t in terms if t in result.params.index]
    if not present:
        return None
    restriction = ", ".join(f"{t} = 0" for t in present)
    try:
        test = result.wald_test(restriction, use_f=True, scalar=True)
        return float(np.squeeze(test.pvalue))
    except Exception:
        return None


def _formula(metric: str, terms: List[str], fixed_effects: bool = True) -> str:
    parts = list(terms)
    if fixed_effects:
        parts += ["C(iso3)", "C(year)"]
    right = " + ".join(parts) if parts else "1"
    return f"{metric} ~ {right}"


# --------------------------------------------------------------------------
# the main entry point
# --------------------------------------------------------------------------

def analyse_metric(panel: Panel, terror_measure: str,
                   lags: List[int]) -> MetricResult:
    """Run the whole model ladder for one metric."""
    if not panel.ok:
        return MetricResult(metric=panel.metric, label=panel.label,
                            unit=panel.unit, terror_measure=terror_measure,
                            n_obs=0, n_countries=0, year_min=0, year_max=0,
                            skipped_reason=panel.skipped_reason)

    data = panel.data
    metric = panel.metric
    controls = panel.control_columns

    result = MetricResult(
        metric=metric, label=panel.label, unit=panel.unit,
        terror_measure=terror_measure, n_obs=len(data),
        n_countries=data["iso3"].nunique(),
        year_min=int(data["year"].min()), year_max=int(data["year"].max()),
        notes=list(panel.notes),
    )

    demeaned = _two_way_demeaned(data[metric], data["iso3"], data["year"])
    total_within = float((demeaned ** 2).sum())

    def build(key: str, terms: List[str]) -> ModelFit:
        formula = _formula(metric, terms)
        fitted = _fit(formula, data)
        rss = float(np.sum(fitted.resid ** 2))
        return ModelFit(
            key=key, name=MODEL_NAMES[key], formula=formula,
            n_obs=int(fitted.nobs), n_countries=data["iso3"].nunique(),
            r2=float(fitted.rsquared), rss=rss,
            r2_within=1.0 - rss / total_within if total_within > 0 else np.nan,
            coefficients=_tidy(fitted), result=fitted,
        )

    # ---- A: before terrorism, B: after terrorism -------------------------
    model_a = build("A", controls)
    model_b = build("B", controls + ["terror"])
    result.models["A"] = model_a
    result.models["B"] = model_b
    result.comparison = _compare(model_a, model_b, data, metric)
    result.control_shifts = _control_shifts(model_a, model_b, controls)

    # ---- C: does the effect differ by group? -----------------------------
    developed_varies = _varies_within_country(data, "developed")
    hetero_terms = controls + ["terror", "terror_x_developed"]
    if developed_varies:
        hetero_terms.insert(-1, "developed")
    else:
        result.notes.append(
            "the 'developed' main effect is absorbed by the country fixed "
            "effects (each country stays in one group), so only the "
            "interaction is estimated - it measures the difference in slope")
    if data["developed"].nunique() > 1:
        model_c = build("C", hetero_terms)
        result.models["C"] = model_c
        result.group_effects = _group_effects(model_c)
    else:
        result.notes.append("only one group is present, so the "
                            "developed/developing comparison was not estimated")

    # ---- D: the long run -------------------------------------------------
    if len(panel.lag_columns) > 1:
        model_d = build("D", controls + panel.lag_columns)
        result.models["D"] = model_d
        result.long_run = _long_run(model_d, panel.lag_columns)
        result.lag_profile = _lag_profile(model_d, panel.lag_columns, lags)

    result.vif = _vif(data, controls + ["terror"])
    result.diagnostics = _diagnostics(model_b, data, metric)
    result.interpretation = _interpret(result, data, terror_measure)
    return result


def _compare(model_a: ModelFit, model_b: ModelFit,
             data: pd.DataFrame, metric: str) -> Dict[str, float]:
    """What the terrorism variable added, on identical rows."""
    terror = model_b.coefficients[model_b.coefficients["term"] == "terror"]
    p_value = _wald_p(model_b.result, ["terror"])
    partial_r2 = ((model_a.rss - model_b.rss) / model_a.rss
                  if model_a.rss > 0 else np.nan)
    return {
        "r2_before": model_a.r2,
        "r2_after": model_b.r2,
        "r2_gain": model_b.r2 - model_a.r2,
        "r2_within_before": model_a.r2_within,
        "r2_within_after": model_b.r2_within,
        "r2_within_gain": model_b.r2_within - model_a.r2_within,
        "partial_r2": partial_r2,
        "terror_coef": float(terror["coef"].iloc[0]) if len(terror) else np.nan,
        "terror_se": float(terror["se"].iloc[0]) if len(terror) else np.nan,
        "terror_ci_low": float(terror["ci_low"].iloc[0]) if len(terror) else np.nan,
        "terror_ci_high": float(terror["ci_high"].iloc[0]) if len(terror) else np.nan,
        "terror_p": p_value if p_value is not None else np.nan,
        "terror_sd": float(data["terror"].std(ddof=1)),
        "metric_sd": float(data[metric].std(ddof=1)),
    }


def _control_shifts(model_a: ModelFit, model_b: ModelFit,
                    controls: List[str]) -> pd.DataFrame:
    """How each control's coefficient moved once terrorism entered."""
    rows = []
    for control in controls:
        before = model_a.coef(control)
        after = model_b.coef(control)
        if before is None or after is None:
            continue
        rows.append({
            "term": control,
            "coef_before": before,
            "coef_after": after,
            "change": after - before,
            "pct_change": (after - before) / abs(before) * 100
            if before not in (0, None) else np.nan,
        })
    return pd.DataFrame(rows)


def _varies_within_country(data: pd.DataFrame, column: str) -> bool:
    return bool((data.groupby("iso3")[column].nunique() > 1).any())


def _group_effects(model_c: ModelFit) -> pd.DataFrame:
    """Terrorism's slope for developing (base) and for developed countries."""
    result = model_c.result
    rows = []

    base = model_c.coefficients[model_c.coefficients["term"] == "terror"]
    if len(base):
        row = base.iloc[0]
        rows.append({"group": "developing", "coef": row["coef"],
                     "se": row["se"], "p": row["p"],
                     "ci_low": row["ci_low"], "ci_high": row["ci_high"]})

    if {"terror", "terror_x_developed"} <= set(result.params.index):
        try:
            test = result.t_test("terror + terror_x_developed = 0")
            conf = np.squeeze(test.conf_int())
            rows.append({"group": "developed",
                         "coef": float(np.squeeze(test.effect)),
                         "se": float(np.squeeze(test.sd)),
                         "p": float(np.squeeze(test.pvalue)),
                         "ci_low": float(conf[0]), "ci_high": float(conf[1])})
        except Exception:
            pass

    interaction = model_c.coefficients[
        model_c.coefficients["term"] == "terror_x_developed"]
    if len(interaction):
        row = interaction.iloc[0]
        rows.append({"group": "difference (developed - developing)",
                     "coef": row["coef"], "se": row["se"], "p": row["p"],
                     "ci_low": row["ci_low"], "ci_high": row["ci_high"]})

    table = pd.DataFrame(rows)
    if not table.empty:
        table["stars"] = table["p"].map(_stars)
    return table


def _long_run(model_d: ModelFit, lag_columns: List[str]) -> Dict[str, float]:
    """The cumulative effect: the sum of the coefficients across all lags."""
    result = model_d.result
    present = [c for c in lag_columns if c in result.params.index]
    if not present:
        return {}
    try:
        test = result.t_test(" + ".join(present) + " = 0")
        conf = np.squeeze(test.conf_int())
        return {
            "cumulative": float(np.squeeze(test.effect)),
            "se": float(np.squeeze(test.sd)),
            "p": float(np.squeeze(test.pvalue)),
            "ci_low": float(conf[0]),
            "ci_high": float(conf[1]),
            "joint_p": _wald_p(result, present) or np.nan,
            "n_lags": len(present),
        }
    except Exception:
        return {}


def _lag_profile(model_d: ModelFit, lag_columns: List[str],
                 lags: List[int]) -> pd.DataFrame:
    """Effect at each lag, plus the running cumulative effect and its CI."""
    result = model_d.result
    rows = []
    for lag, column in zip(lags, lag_columns):
        row = model_d.coefficients[model_d.coefficients["term"] == column]
        if row.empty:
            continue
        row = row.iloc[0]
        so_far = [c for c in lag_columns[:lag_columns.index(column) + 1]
                  if c in result.params.index]
        cumulative = cumulative_low = cumulative_high = np.nan
        try:
            test = result.t_test(" + ".join(so_far) + " = 0")
            conf = np.squeeze(test.conf_int())
            cumulative = float(np.squeeze(test.effect))
            cumulative_low, cumulative_high = float(conf[0]), float(conf[1])
        except Exception:
            pass
        rows.append({
            "lag": lag, "term": column,
            "coef": row["coef"], "se": row["se"], "p": row["p"],
            "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "cumulative": cumulative,
            "cumulative_ci_low": cumulative_low,
            "cumulative_ci_high": cumulative_high,
        })
    return pd.DataFrame(rows)


def _vif(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Multicollinearity among the regressors that are not fixed effects."""
    usable = [c for c in columns if c in data.columns
              and data[c].std(ddof=0) > 0]
    if len(usable) < 2:
        return pd.DataFrame()
    matrix = data[usable].astype(float).copy()
    matrix.insert(0, "const", 1.0)
    rows = []
    for position, column in enumerate(matrix.columns):
        if column == "const":
            continue
        try:
            value = variance_inflation_factor(matrix.values, position)
        except Exception:
            value = np.nan
        rows.append({"term": column, "vif": value})
    return pd.DataFrame(rows)


def _diagnostics(model_b: ModelFit, data: pd.DataFrame,
                 metric: str) -> Dict[str, float]:
    result = model_b.result
    country_terms = [t for t in result.params.index if t.startswith("C(iso3)")]
    year_terms = [t for t in result.params.index if t.startswith("C(year)")]
    return {
        "n_obs": float(len(data)),
        "n_countries": float(data["iso3"].nunique()),
        "n_years": float(data["year"].nunique()),
        "country_fe_p": _wald_p(result, country_terms) or np.nan,
        "year_fe_p": _wald_p(result, year_terms) or np.nan,
        "metric_mean": float(data[metric].mean()),
        "metric_sd": float(data[metric].std(ddof=1)),
        "terror_mean": float(data["terror"].mean()),
        "terror_share_nonzero": float((data["terror"] > 0).mean()),
    }


def _interpret(result: MetricResult, data: pd.DataFrame,
               terror_measure: str) -> List[str]:
    """Plain-language readings of the numbers, for the report."""
    lines = []
    comparison = result.comparison
    coefficient = comparison.get("terror_coef", np.nan)
    p_value = comparison.get("terror_p", np.nan)
    unit = result.unit or "units"

    if not np.isfinite(coefficient):
        return lines

    direction = "lower" if coefficient < 0 else "higher"
    significance = ("statistically significant at 5%"
                    if np.isfinite(p_value) and p_value < 0.05
                    else "not statistically significant at 5%")

    if terror_measure.startswith("log"):
        per_tenfold = coefficient * math.log(10)
        lines.append(
            f"A tenfold increase in {terror_measure.replace('log_', '')} is "
            f"associated with {abs(per_tenfold):.3g} {unit} {direction} "
            f"{result.label} ({significance}).")
    else:
        lines.append(
            f"One more unit of {terror_measure} is associated with "
            f"{abs(coefficient):.3g} {unit} {direction} {result.label} "
            f"({significance}).")

    terror_sd = comparison.get("terror_sd", np.nan)
    if np.isfinite(terror_sd) and terror_sd > 0:
        lines.append(
            f"A one standard deviation increase in terrorism "
            f"({terror_sd:.2f}) moves {result.label} by "
            f"{coefficient * terror_sd:+.3g} {unit}.")

    gain = comparison.get("r2_within_gain", np.nan)
    if np.isfinite(gain):
        lines.append(
            f"Adding terrorism raised the within-country R2 from "
            f"{comparison['r2_within_before']:.3f} to "
            f"{comparison['r2_within_after']:.3f} "
            f"(+{gain:.4f}).")

    if not result.group_effects.empty:
        table = result.group_effects.set_index("group")
        if "difference (developed - developing)" in table.index:
            row = table.loc["difference (developed - developing)"]
            verdict = ("differs significantly" if row["p"] < 0.05
                       else "does not differ significantly")
            lines.append(
                f"The effect in developed countries {verdict} from developing "
                f"countries (difference {row['coef']:+.3g}, p={row['p']:.3f}).")

    if result.long_run:
        long_run = result.long_run
        lines.append(
            f"Cumulated over {long_run['n_lags']} years the effect is "
            f"{long_run['cumulative']:+.3g} {unit} "
            f"(p={long_run['p']:.3f}), against {coefficient:+.3g} in the "
            f"first year alone.")

    return lines


def partial_regression_frame(panel: Panel) -> pd.DataFrame:
    """Data for the added-variable plot: the isolated terrorism effect.

    Both the metric and terrorism are stripped of everything the controls and
    the fixed effects already explain; what is left is the relationship the
    coefficient actually measures. The slope through these residuals equals the
    regression coefficient on terrorism.
    """
    data = panel.data
    controls = panel.control_columns

    metric_resid = _fit(_formula(panel.metric, controls), data).resid
    terror_resid = _fit(_formula("terror", controls), data).resid

    return pd.DataFrame({
        "iso3": data["iso3"].values,
        "year": data["year"].values,
        "group": data["group"].values,
        "terror_resid": np.asarray(terror_resid),
        "metric_resid": np.asarray(metric_resid),
    })


def run_all(panels: Dict[str, Panel], terror_measure: str,
            lags: List[int]) -> Dict[str, MetricResult]:
    """Analyse every metric independently. A failure is recorded, not raised."""
    results = {}
    for name, panel in panels.items():
        if not panel.ok:
            results[name] = analyse_metric(panel, terror_measure, lags)
            print(f"  {name}: SKIPPED - {panel.skipped_reason}")
            continue
        try:
            results[name] = analyse_metric(panel, terror_measure, lags)
            comparison = results[name].comparison
            print(f"  {name}: terror coef {comparison['terror_coef']:+.4g} "
                  f"(p={comparison['terror_p']:.3f}), "
                  f"within-R2 {comparison['r2_within_before']:.3f} -> "
                  f"{comparison['r2_within_after']:.3f}")
        except Exception as error:                       # keep the run alive
            results[name] = MetricResult(
                metric=name, label=panel.label, unit=panel.unit,
                terror_measure=terror_measure, n_obs=panel.n_obs,
                n_countries=panel.n_countries, year_min=0, year_max=0,
                skipped_reason=f"regression failed: {error}")
            print(f"  {name}: FAILED - {error}")
    return results
