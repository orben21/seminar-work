"""A known answer, planted on the real panel, to prove the regressions work.

3_run_analysis.py --sample keeps the real markets, years, groups, TSI and
controls, and replaces the two outcomes with synthetic ones:

    planted_outcome  = market effect + year effect
                       + TRUE_DEVELOPING x TSI(t)        in developing markets
                       + TRUE_DEVELOPED  x TSI(t)        in developed markets
                       + TRUE_LAG x TSI(t-1) + controls + noise
    placebo_outcome  = the same with every TSI term removed

The four regressions of §7.1 must recover the planted numbers - in particular
b3 = TRUE_DEVELOPED - TRUE_DEVELOPING, the H1/H2 test - and find nothing in the
placebo. A single draw never lands exactly on the truth, so "recovered" means
within RECOVERY_TOLERANCE_SE standard errors.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .panel import TREATMENT, TREATMENT_LAG, INTERACTION, INTERACTION_LAG, Panel
from .specs import OutcomeSpec

TRUE_DEVELOPING = -0.80      # effect of a 1 SD rise in the TSI, developing
TRUE_DEVELOPED = -0.20       # ... and in developed markets
TRUE_LAG = -0.30             # the same one-year-later effect in both groups
NOISE_SD = 1.0
RANDOM_SEED = 20260918
RECOVERY_TOLERANCE_SE = 2.5

PLANTED = "planted_outcome"
PLACEBO = "placebo_outcome"


def plant(panel: Panel, seed: int = RANDOM_SEED) -> Panel:
    rng = np.random.default_rng(seed)
    data = panel.data.copy()
    markets = sorted(data["iso3"].unique())
    years = sorted(data["year"].unique())
    market_effect = dict(zip(markets, rng.normal(0, 2.0, len(markets))))
    year_effect = dict(zip(years, rng.normal(0, 1.0, len(years))))

    base = (data["iso3"].map(market_effect) + data["year"].map(year_effect)
            + rng.normal(0, NOISE_SD, len(data)))
    for control in panel.control_names:
        column = data[control].astype(float)
        spread = column.std()
        if spread and np.isfinite(spread):
            base = base + rng.normal(0, 0.3) * (column - column.mean()) / spread
    slope = np.where(data["developed"] == 1.0, TRUE_DEVELOPED,
                     TRUE_DEVELOPING)
    data[PLANTED] = base + slope * data[TREATMENT] \
        + TRUE_LAG * data[TREATMENT_LAG]
    data[PLACEBO] = base

    outcomes = {
        PLANTED: OutcomeSpec(files=("synthetic",), hypothesis="H1",
                             label="Planted outcome", unit="points"),
        PLACEBO: OutcomeSpec(files=("synthetic",), hypothesis="H2",
                             label="Placebo outcome", unit="points"),
    }
    return Panel(data=data, outcomes=outcomes, controls=panel.controls,
                 notes=["SYNTHETIC outcomes with a planted TSI effect"])


def _verdict(estimate: float, se: float, truth: float) -> str:
    if not se or not np.isfinite(se):
        return "no standard error"
    distance = abs(estimate - truth) / se
    mark = "RECOVERED" if distance <= RECOVERY_TOLERANCE_SE else "OFF"
    return f"{distance:.1f} SE away -> {mark}"


def check_recovery(core: Dict[str, Dict[str, object]]) -> List[str]:
    """Readable lines: planted value, estimate, verdict."""
    lines = []
    fits = core.get(PLANTED, {})
    checks = [
        ("4_interaction", TREATMENT, TRUE_DEVELOPING,
         "b1  TSI(t), developing"),
        ("4_interaction", TREATMENT_LAG, TRUE_LAG, "b2  TSI(t-1)"),
        ("4_interaction", INTERACTION, TRUE_DEVELOPED - TRUE_DEVELOPING,
         "b3  TSI(t) x Developed"),
        ("4_interaction", INTERACTION_LAG, 0.0, "b4  TSI(t-1) x Developed"),
        ("2_developed", TREATMENT, TRUE_DEVELOPED, "regression 2, TSI(t)"),
        ("3_developing", TREATMENT, TRUE_DEVELOPING, "regression 3, TSI(t)"),
    ]
    for key, term, truth, label in checks:
        fit = fits.get(key)
        row = fit.get(term) if fit is not None else None
        if row is None:
            lines.append(f"{label:<28} not estimated")
            continue
        lines.append(f"{label:<28} planted {truth:+.3f}   estimate "
                     f"{row['coef']:+.3f} (SE {row['se']:.3f})   "
                     f"{_verdict(row['coef'], row['se'], truth)}")
    placebo = core.get(PLACEBO, {}).get("4_interaction")
    if placebo is not None:
        for term in (TREATMENT, INTERACTION):
            row = placebo.get(term)
            inside = row["ci_low"] <= 0 <= row["ci_high"]
            lines.append(f"placebo {term:<20} estimate {row['coef']:+.3f}, 95% "
                         f"CI [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] "
                         f"{'contains 0 as it should' if inside else 'EXCLUDES 0'}")
    return lines
