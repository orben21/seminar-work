"""Figures. Everything is saved as a 300 dpi PNG, ready to drop into the paper.

Seven kinds of figure:

  1 coverage heatmap        what data you actually have, per metric
  2 before/after chart      Model A next to Model B - what terrorism changed
  3 effect forest plot      the terror effect, developed vs developing
  4 lag profile             the cumulative long-run effect year by year
  5 partial regression      the isolated terror effect as a scatter
  6 group trends            metric over time by group, terrorism overlaid
  7 cross-metric summary    every metric's terror effect on one comparable axis
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")                      # no display needed; write files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .panel import Panel
from .regression import MetricResult, _stars

# One consistent look for every figure.
COLOR_BEFORE = "#8895a7"
COLOR_AFTER = "#1f4e79"
COLOR_TERROR = "#b23a48"
COLOR_DEVELOPED = "#1f4e79"
COLOR_DEVELOPING = "#d98032"
COLOR_GRID = "#dfe3e8"
GROUP_COLORS = {"developed": COLOR_DEVELOPED, "developing": COLOR_DEVELOPING}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#5b6672",
    "axes.labelcolor": "#22282e",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "font.size": 10,
})


def _style(axis, title="", xlabel="", ylabel=""):
    axis.set_title(title, loc="left", pad=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color=COLOR_GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    return axis


def _save(figure, path: Path, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# 1. data coverage
# --------------------------------------------------------------------------

def coverage_heatmap(matrix: pd.DataFrame, name: str, path: Path,
                     max_countries: int = 60, dpi: int = 300) -> Optional[Path]:
    """Which country-years actually have data. Look at this first."""
    if matrix.empty:
        return None
    ordered = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    truncated = len(ordered) > max_countries
    shown = ordered.head(max_countries)

    height = max(3.0, 0.16 * len(shown) + 1.6)
    figure, axis = plt.subplots(figsize=(11, height))
    axis.imshow(shown.values, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                interpolation="nearest")

    axis.set_yticks(range(len(shown)))
    axis.set_yticklabels(shown.index, fontsize=6)
    years = list(shown.columns)
    step = max(1, len(years) // 16)
    axis.set_xticks(range(0, len(years), step))
    axis.set_xticklabels(years[::step], rotation=45, ha="right")

    filled = float(shown.values.mean()) * 100
    title = f"Data coverage: {name}  ({filled:.0f}% of country-years present)"
    if truncated:
        title += f"  [top {max_countries} of {len(ordered)} countries]"
    axis.set_title(title, loc="left", pad=10)
    axis.grid(False)
    return _save(figure, path, dpi)


# --------------------------------------------------------------------------
# 2. before / after terrorism
# --------------------------------------------------------------------------

def before_after_chart(result: MetricResult, path: Path,
                       dpi: int = 300) -> Optional[Path]:
    """Every coefficient in Model A next to Model B, plus the R2 gain."""
    if "A" not in result.models or "B" not in result.models:
        return None
    model_a, model_b = result.models["A"], result.models["B"]

    def rows(model):
        table = model.coefficients
        return table[table["term"] != "Intercept"].set_index("term")

    before, after = rows(model_a), rows(model_b)
    controls = [t for t in after.index if t in before.index]
    added = [t for t in after.index if t not in before.index]
    if not controls and not added:
        return None

    # The terrorism term gets its own x-axis. Its confidence interval is usually
    # far narrower than the controls', and on a shared scale it collapses into
    # a dot right when it is the thing you came to look at.
    height = 1.4 + 0.62 * (len(controls) + len(added))
    figure = plt.figure(figsize=(11, height))
    grid = figure.add_gridspec(
        2, 2, width_ratios=[3, 1],
        height_ratios=[max(len(controls), 1), max(len(added), 1)],
        hspace=0.55, wspace=0.08)
    axis = figure.add_subplot(grid[0, 0])
    lower = figure.add_subplot(grid[1, 0])
    panel = figure.add_subplot(grid[:, 1])

    def draw(target, terms, show_before):
        positions = np.arange(len(terms))
        offset = 0.18 if show_before else 0.0
        for index, term in enumerate(terms):
            if show_before and term in before.index:
                row = before.loc[term]
                target.errorbar(row["coef"], positions[index] + offset,
                                xerr=[[row["coef"] - row["ci_low"]],
                                      [row["ci_high"] - row["coef"]]],
                                fmt="o", color=COLOR_BEFORE, capsize=3,
                                markersize=6,
                                label="before terrorism" if index == 0 else None)
            row = after.loc[term]
            color = COLOR_TERROR if term == "terror" else COLOR_AFTER
            target.errorbar(row["coef"], positions[index] - offset,
                            xerr=[[row["coef"] - row["ci_low"]],
                                  [row["ci_high"] - row["coef"]]],
                            fmt="o", color=color, capsize=3, markersize=6,
                            label="after terrorism" if index == 0 else None)
        target.axvline(0, color="#22282e", linewidth=1.0, linestyle="--",
                       zorder=1)
        target.set_yticks(positions)
        target.set_yticklabels(terms)
        target.invert_yaxis()
        target.margins(y=0.35)

    draw(axis, controls, show_before=True)
    _style(axis, f"{result.label}: coefficients before and after adding terrorism",
           f"controls - coefficient (95% CI), {result.unit or 'metric units'}")

    draw(lower, added, show_before=False)
    _style(lower, "", f"terrorism - coefficient (95% CI), own scale, "
                      f"{result.unit or 'metric units'}")

    comparison = result.comparison
    panel.axis("off")
    lines = [
        "within-country R2",
        f"  before   {comparison['r2_within_before']:.4f}",
        f"  after    {comparison['r2_within_after']:.4f}",
        f"  gain    {comparison['r2_within_gain']:+.4f}",
        "",
        "terrorism term",
        f"  coef  {comparison['terror_coef']:+.4g}",
        f"  p     {comparison['terror_p']:.4f}",
        "",
        f"{result.n_obs:,} country-years",
        f"{result.n_countries} countries",
        f"{result.year_min}-{result.year_max}",
    ]
    panel.text(0.0, 0.98, "\n".join(lines), va="top", ha="left",
               family="monospace", fontsize=9)
    # The legend lives here, where there is room: on the plot itself it sits on
    # top of the error bars.
    handles = [plt.Line2D([], [], color=COLOR_BEFORE, marker="o", linestyle="none"),
               plt.Line2D([], [], color=COLOR_AFTER, marker="o", linestyle="none"),
               plt.Line2D([], [], color=COLOR_TERROR, marker="o", linestyle="none")]
    panel.legend(handles, ["before terrorism", "after terrorism", "terrorism term"],
                 loc="lower left", fontsize=9, handletextpad=0.4)
    return _save(figure, path, dpi)


# --------------------------------------------------------------------------
# 3. the effect, developed vs developing
# --------------------------------------------------------------------------

def effect_forest(result: MetricResult, path: Path,
                  dpi: int = 300) -> Optional[Path]:
    """Terrorism's estimated effect overall and by group."""
    rows = []
    comparison = result.comparison
    if np.isfinite(comparison.get("terror_coef", np.nan)):
        rows.append({"group": "all countries",
                     "coef": comparison["terror_coef"],
                     "ci_low": comparison["terror_ci_low"],
                     "ci_high": comparison["terror_ci_high"],
                     "p": comparison["terror_p"]})
    if not result.group_effects.empty:
        rows.extend(result.group_effects.to_dict("records"))
    if not rows:
        return None

    table = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(8.5, 1.2 + 0.7 * len(table)))
    positions = np.arange(len(table))

    for index, row in table.iterrows():
        color = GROUP_COLORS.get(str(row["group"]), COLOR_AFTER)
        if str(row["group"]).startswith("difference"):
            color = COLOR_TERROR
        axis.errorbar(row["coef"], positions[index],
                      xerr=[[row["coef"] - row["ci_low"]],
                            [row["ci_high"] - row["coef"]]],
                      fmt="o", color=color, capsize=4, markersize=8)
        stars = _stars(row["p"])          # never trust a missing 'stars' cell
        axis.annotate(f"{row['coef']:+.3g}{stars}",
                      (row["coef"], positions[index]),
                      textcoords="offset points", xytext=(0, 11),
                      ha="center", fontsize=9, color=color)

    axis.axvline(0, color="#22282e", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels(table["group"])
    axis.invert_yaxis()
    axis.margins(y=0.25)
    _style(axis,
           f"{result.label}: effect of {result.terror_measure}",
           f"change in {result.label} ({result.unit or 'units'}) per unit of "
           f"{result.terror_measure}")
    return _save(figure, path, dpi)


# --------------------------------------------------------------------------
# 4. the long run
# --------------------------------------------------------------------------

def lag_profile_chart(result: MetricResult, path: Path,
                      dpi: int = 300) -> Optional[Path]:
    """Effect at each lag, and the cumulative effect with its confidence band."""
    profile = result.lag_profile
    if profile.empty:
        return None

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(profile["lag"], profile["coef"], width=0.45,
             color=COLOR_BEFORE, label="effect in that year alone", zorder=2)
    axis.errorbar(profile["lag"], profile["coef"],
                  yerr=[profile["coef"] - profile["ci_low"],
                        profile["ci_high"] - profile["coef"]],
                  fmt="none", ecolor="#5b6672", capsize=3, zorder=3)

    if profile["cumulative"].notna().any():
        axis.plot(profile["lag"], profile["cumulative"], "-o",
                  color=COLOR_TERROR, linewidth=2, markersize=6,
                  label="cumulative effect", zorder=4)
        axis.fill_between(profile["lag"], profile["cumulative_ci_low"],
                          profile["cumulative_ci_high"],
                          color=COLOR_TERROR, alpha=0.15, zorder=1)

    axis.axhline(0, color="#22282e", linewidth=1.0, linestyle="--")
    axis.set_xticks(profile["lag"])
    axis.legend(loc="best")
    _style(axis,
           f"{result.label}: how long the effect of terrorism lasts",
           "years since the terrorism happened",
           f"change in {result.label} ({result.unit or 'units'})")
    return _save(figure, path, dpi)


# --------------------------------------------------------------------------
# 5. the isolated effect
# --------------------------------------------------------------------------

def partial_regression_plot(frame: pd.DataFrame, result: MetricResult,
                            path: Path, dpi: int = 300) -> Optional[Path]:
    """Terrorism vs the metric once everything else has been partialled out."""
    if frame.empty:
        return None

    figure, axis = plt.subplots(figsize=(8.5, 6))
    for group, chunk in frame.groupby("group", dropna=True):
        axis.scatter(chunk["terror_resid"], chunk["metric_resid"], s=12,
                     alpha=0.45, label=str(group), zorder=2,
                     color=GROUP_COLORS.get(str(group), COLOR_AFTER))

    x = frame["terror_resid"].to_numpy(dtype=float)
    y = frame["metric_resid"].to_numpy(dtype=float)
    if len(x) > 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        line = np.linspace(x.min(), x.max(), 100)
        axis.plot(line, slope * line + intercept, color=COLOR_TERROR,
                  linewidth=2, zorder=3,
                  label=f"slope = {slope:+.3g} (the coefficient)")

    axis.axhline(0, color="#9aa4ae", linewidth=0.8)
    axis.axvline(0, color="#9aa4ae", linewidth=0.8)

    # A handful of extreme country-years can squash the whole cloud into a line.
    # The view is clipped to the central 99%, and says how many points that hides
    # - the fitted line still uses every observation.
    low, high = np.nanpercentile(y, [0.5, 99.5])
    padding = 0.08 * (high - low) if high > low else 1.0
    hidden = int(((y < low - padding) | (y > high + padding)).sum())
    if hidden:
        axis.set_ylim(low - padding, high + padding)
        axis.annotate(f"{hidden} extreme point(s) outside this view;\n"
                      f"the fitted line uses all {len(y):,} observations",
                      xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
                      color="#667180", va="bottom")

    axis.legend(loc="best")
    _style(axis,
           f"{result.label}: the isolated effect of terrorism",
           f"{result.terror_measure}, with controls and fixed effects removed",
           f"{result.label}, with controls and fixed effects removed")
    return _save(figure, path, dpi)


# --------------------------------------------------------------------------
# 6. group trends over time
# --------------------------------------------------------------------------

def group_trend_chart(panel: Panel, result: MetricResult, path: Path,
                      dpi: int = 300) -> Optional[Path]:
    """The metric over time for each group, with terrorism intensity behind it."""
    data = panel.data
    if data.empty:
        return None

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for group, chunk in data.groupby("group", dropna=True):
        yearly = chunk.groupby("year")[panel.metric].mean()
        axis.plot(yearly.index, yearly.values, "-o", markersize=4, linewidth=2,
                  color=GROUP_COLORS.get(str(group), COLOR_AFTER),
                  label=f"{group} (mean {panel.metric})", zorder=3)

    _style(axis, f"{result.label} and terrorism over time", "year",
           f"{result.label} ({result.unit or 'units'})")

    twin = axis.twinx()
    terror_by_year = data.groupby("year")["terror"].mean()
    twin.fill_between(terror_by_year.index, terror_by_year.values, 0,
                      color=COLOR_TERROR, alpha=0.12, zorder=1)
    twin.set_ylabel(f"mean {result.terror_measure}", color=COLOR_TERROR)
    twin.tick_params(axis="y", colors=COLOR_TERROR)
    twin.spines["top"].set_visible(False)
    twin.grid(False)

    handles, labels = axis.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_TERROR, alpha=0.12))
    labels.append(f"mean {result.terror_measure} (right axis)")
    axis.legend(handles, labels, loc="best")
    return _save(figure, path, dpi)


# --------------------------------------------------------------------------
# 7. every metric on one axis
# --------------------------------------------------------------------------

def cross_metric_summary(results: Dict[str, MetricResult], path: Path,
                         dpi: int = 300) -> Optional[Path]:
    """Standardised terror effect per metric: comparable across every metric.

    Each coefficient is rescaled to "standard deviations of the metric per one
    standard deviation of terrorism", so metrics measured in different units sit
    on the same axis.
    """
    rows = []
    for name, result in results.items():
        if not result.ok or not result.comparison:
            continue
        comparison = result.comparison
        metric_sd = comparison.get("metric_sd", np.nan)
        terror_sd = comparison.get("terror_sd", np.nan)
        if not (np.isfinite(metric_sd) and metric_sd > 0):
            continue
        scale = terror_sd / metric_sd
        rows.append({
            "metric": result.label or name,
            "beta": comparison["terror_coef"] * scale,
            "ci_low": comparison["terror_ci_low"] * scale,
            "ci_high": comparison["terror_ci_high"] * scale,
            "p": comparison["terror_p"],
        })
    if not rows:
        return None

    table = pd.DataFrame(rows).sort_values("beta")
    figure, axis = plt.subplots(figsize=(9, 1.6 + 0.7 * len(table)))
    positions = np.arange(len(table))

    for index, (_, row) in enumerate(table.iterrows()):
        significant = np.isfinite(row["p"]) and row["p"] < 0.05
        color = COLOR_TERROR if significant else "#9aa4ae"
        axis.errorbar(row["beta"], positions[index],
                      xerr=[[row["beta"] - row["ci_low"]],
                            [row["ci_high"] - row["beta"]]],
                      fmt="o", color=color, capsize=4, markersize=8)
        axis.annotate(f"{row['beta']:+.3f}" + ("*" if significant else ""),
                      (row["beta"], positions[index]),
                      textcoords="offset points", xytext=(0, 11), ha="center",
                      fontsize=9, color=color)

    axis.axvline(0, color="#22282e", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels(table["metric"])
    axis.margins(y=0.2)
    _style(axis, "Effect of terrorism across every metric",
           "standard deviations of the metric per 1 SD of terrorism\n"
           "(filled red = significant at 5%)")
    return _save(figure, path, dpi)


def terror_overview(country_year: pd.DataFrame, path: Path,
                    dpi: int = 300) -> Optional[Path]:
    """Global terrorism over time - context for the paper's opening."""
    if country_year.empty:
        return None
    yearly = country_year.groupby("year")[["n_attacks", "n_killed"]].sum()

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(yearly.index, yearly["n_attacks"], "-o", markersize=4,
              color=COLOR_AFTER, linewidth=2, label="attacks")
    _style(axis, "Terrorism worldwide, as recorded by the GTD", "year",
           "attacks per year")

    twin = axis.twinx()
    twin.plot(yearly.index, yearly["n_killed"], "-", color=COLOR_TERROR,
              linewidth=2, label="people killed")
    twin.set_ylabel("people killed per year", color=COLOR_TERROR)
    twin.tick_params(axis="y", colors=COLOR_TERROR)
    twin.spines["top"].set_visible(False)
    twin.grid(False)

    handles = [plt.Line2D([], [], color=COLOR_AFTER, marker="o"),
               plt.Line2D([], [], color=COLOR_TERROR)]
    axis.legend(handles, ["attacks (left)", "people killed (right)"],
                loc="upper left")
    return _save(figure, path, dpi)
