"""The data figures of the plan's master list (Part 3), numbered as there.

    1   qualifying attacks by group, 1998-2020          §1.1
    5   sectoral composition of the sample              §3.3.2
    6   sample selection funnel Q1-Q6                   §3.5
    8   HDI distribution with the 0.800 cut-off         §4.2
    9   HDI against GNI per capita                      §4.3
    10  cross-institutional agreement matrix            §4.4
    11  event qualification funnel K1-K6 + floor        §5.2
    12  composition of qualifying events                §5.4
    13A skewness of annual fatality counts, by market   §5.6
    14  raw vs log vs z-score                           §5.6
    15  TSI heatmap, market x year                      §6.6
    16  correlation matrix                              §6.6
    17  TSI components over time by group               §8.1
    18  TSI distribution by group                       §8.1
    19  GDP growth and FDI ratio over time              §8.2
    20  coefficient plot, GDP growth                    §8.4
    21  coefficient plot, FDI ratio                     §8.5
    22  marginal effect of TSI by group (headline)      §8.6
    23  binned scatter, TSI vs outcome                  §8.6
    24  component coefficients                          §8.7

Figures 2, 3, 4, 7 and 13 are conceptual diagrams and are not drawn from data
(Figure 13's worked example is written as t5_6_worked_example.csv).

Consistency rules (Part 3): one colour per group in every figure, every axis
labelled with units, every figure captioned with its data source.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")                      # no display needed; write files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .tsi import COMPONENT_LABELS, COMPONENTS

COLOR_DEVELOPED = "#1f4e79"
COLOR_DEVELOPING = "#d98032"
COLOR_NEUTRAL = "#8895a7"
COLOR_ACCENT = "#b23a48"
COLOR_GRID = "#dfe3e8"
COLOR_TEXT = "#22282e"
GROUP_COLORS = {"developed": COLOR_DEVELOPED, "developing": COLOR_DEVELOPING}
GROUPS = ("developed", "developing")

GTD_SOURCE = "GTD 0522 (START); qualifying events, File 3 K1-K6 + floor"
WDI_SOURCE = "World Bank WDI"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#5b6672", "axes.labelcolor": COLOR_TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
    "font.size": 10,
})


def _style(axis, title="", xlabel="", ylabel="", grid_axis="both"):
    axis.set_title(title, loc="left", pad=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis=grid_axis, color=COLOR_GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    return axis


def _save(figure, path: Path, source: str, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.text(0.01, 0.008, f"Source: {source}", fontsize=7.5,
                color="#667180", ha="left", va="bottom")
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _group_of(sample: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(sample["iso3"], sample["group"]))


def _labelled(table: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Attach country name and group from the classified sample, replacing
    any copies the table already carries."""
    table = table.drop(columns=[c for c in ("country", "group")
                                if c in table.columns])
    return table.merge(sample[["iso3", "country", "group"]], on="iso3")


def _colour_ticklabels(axis, labels: Sequence[str], groups: Sequence[str]):
    for tick, group in zip(axis.get_yticklabels(), groups):
        tick.set_color(GROUP_COLORS.get(group, COLOR_TEXT))


def _group_legend(axis, loc="best", extra: Optional[List] = None):
    handles = [Patch(color=GROUP_COLORS[g], label=g) for g in GROUPS]
    axis.legend(handles=handles + (extra or []), loc=loc)


# --------------------------------------------------------------------------
# 1  qualifying attacks by group
# --------------------------------------------------------------------------

def fig01_qualifying_attacks(country_year: pd.DataFrame, sample: pd.DataFrame,
                             path: Path, dpi: int = 300) -> Path:
    data = country_year.assign(group=country_year["iso3"].map(
        _group_of(sample)))
    yearly = data.pivot_table(index="year", columns="group", values="NA",
                              aggfunc="sum")
    figure, axis = plt.subplots(figsize=(10, 5))
    for year, label in ((2008, "Apr 2008"), (2012, "Jan 2012")):
        axis.axvspan(year - 0.5, year + 0.5, color="#eceff3", zorder=0)
        axis.text(year, 1.0, f"GTD collection\nbreak, {label}",
                  transform=axis.get_xaxis_transform(), ha="center",
                  va="top", fontsize=8, color="#667180")
    for group in GROUPS:
        if group in yearly.columns:
            axis.plot(yearly.index, yearly[group], "-o", markersize=4,
                      linewidth=2, color=GROUP_COLORS[group],
                      label=f"{group} markets")
    axis.legend(loc="upper left")
    _style(axis, "Figure 1 - Qualifying terrorist attacks, 1998-2020",
           "year", "qualifying attacks per year (count)")
    return _save(figure, path, GTD_SOURCE + "; sample markets of File 1", dpi)


# --------------------------------------------------------------------------
# 5  sectoral composition
# --------------------------------------------------------------------------

def fig05_sectoral_composition(shares: pd.DataFrame, sample: pd.DataFrame,
                               base_year: int, path: Path,
                               dpi: int = 300) -> Optional[Path]:
    data = sample[["iso3", "country", "group", "hdi"]].merge(
        shares, on="iso3", how="inner").dropna(
        subset=["agriculture", "industry", "services"])
    if data.empty:
        return None
    parts = data[["agriculture", "industry", "services"]]
    data[["agriculture", "industry", "services"]] = parts.div(
        parts.sum(axis=1), axis=0) * 100
    data = data.sort_values("hdi", ascending=True)
    figure, axis = plt.subplots(figsize=(9.5, max(4, 0.19 * len(data) + 1.5)))
    left = np.zeros(len(data))
    colors = {"agriculture": "#9bbf85", "industry": "#8895a7",
              "services": "#c9d4e0"}
    for sector in ("agriculture", "industry", "services"):
        axis.barh(data["country"], data[sector], left=left,
                  color=colors[sector], label=sector, height=0.8)
        left += data[sector].to_numpy()
    axis.set_xlim(0, 100)
    axis.tick_params(axis="y", labelsize=7)
    _colour_ticklabels(axis, data["country"], data["group"])
    axis.legend(loc="lower right", ncol=3, fontsize=8)
    _style(axis, f"Figure 5 - Sectoral composition at the base year "
                 f"({base_year}), sorted by HDI",
           "share of value added (%, rescaled to sum to 100)", "",
           grid_axis="x")
    return _save(figure, path, f"{WDI_SOURCE} NV.AGR/NV.IND/NV.SRV.TOTL.ZS; "
                 f"labels coloured by group (blue developed, orange "
                 f"developing)", dpi)


# --------------------------------------------------------------------------
# 6  sample selection funnel
# --------------------------------------------------------------------------

def fig06_sample_funnel(funnel: pd.DataFrame, path: Path,
                        dpi: int = 300) -> Path:
    import textwrap
    figure, axis = plt.subplots(figsize=(11, 5.6))
    positions = np.arange(len(funnel))
    axis.bar(positions, funnel["remaining"], color=COLOR_DEVELOPED,
             width=0.62, zorder=2)
    axis.bar(positions, funnel["dropped"], bottom=funnel["remaining"],
             color="#e4c1c6", hatch="//", edgecolor="white", width=0.62,
             zorder=2, label="dropped at this rule")
    for x, (_, row) in zip(positions, funnel.iterrows()):
        axis.text(x, row["remaining"] / 2, f"{int(row['remaining'])}",
                  ha="center", va="center", color="white", fontweight="bold")
        if row["dropped"]:
            axis.text(x, row["remaining"] + row["dropped"] + 1,
                      f"-{int(row['dropped'])}", ha="center", va="bottom",
                      color=COLOR_ACCENT, fontsize=9)
    labels = [textwrap.fill(str(label), 16) for label in funnel["label"]]
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylim(0, (funnel["remaining"] + funnel["dropped"]).max() * 1.12)
    axis.legend(loc="upper right")
    _style(axis, "Figure 6 - Sample selection funnel (File 1, Q1-Q6)", "",
           "markets (count)", grid_axis="y")
    return _save(figure, path, "GTD 0522; World Bank WDI; File 1 rules - "
                 "exclusion log in t3_5_exclusion_log.csv", dpi)


# --------------------------------------------------------------------------
# 8  HDI distribution
# --------------------------------------------------------------------------

def fig08_hdi_distribution(sample: pd.DataFrame, base_year: int, path: Path,
                           dpi: int = 300) -> Path:
    data = sample.dropna(subset=["hdi"]).sort_values("hdi", ascending=True)
    figure, axis = plt.subplots(figsize=(8.5, max(4, 0.19 * len(data) + 1.5)))
    axis.axvspan(0.78, 0.82, color="#f3e7c9", zorder=0,
                 label="borderline band 0.780-0.820")
    colors = data["group"].map(GROUP_COLORS).fillna(COLOR_NEUTRAL)
    labels = [f"{c} *" if b else c for c, b in zip(data["country"],
                                                   data["borderline"])]
    axis.barh(labels, data["hdi"], color=colors, height=0.75, zorder=2)
    axis.axvline(0.8, color=COLOR_TEXT, linestyle="--", linewidth=1.2,
                 zorder=3)
    axis.set_xlim(max(0.0, data["hdi"].min() - 0.05), 1.0)
    axis.tick_params(axis="y", labelsize=7)
    _group_legend(axis, "lower right",
                  [Patch(color="#f3e7c9", label="borderline band")])
    _style(axis, f"Figure 8 - HDI of the sample at the base year ({base_year})",
           "Human Development Index (0-1); dashed line = 0.800 (C1)", "",
           grid_axis="x")
    return _save(figure, path, "UNDP HDR composite time series; * = flagged "
                 "borderline under File 2 §6.2 (any trigger)", dpi)


# --------------------------------------------------------------------------
# 9  HDI vs GNI per capita
# --------------------------------------------------------------------------

def fig09_hdi_vs_gni(sample: pd.DataFrame, threshold: float, base_year: int,
                     path: Path, dpi: int = 300) -> Path:
    data = sample.dropna(subset=["hdi", "gni_atlas"])
    figure, axis = plt.subplots(figsize=(9, 6.5))
    for group in GROUPS:
        chunk = data[data["group"] == group]
        axis.scatter(chunk["gni_atlas"], chunk["hdi"], s=36,
                     color=GROUP_COLORS[group], label=f"{group} (C1)",
                     zorder=3, edgecolor="white", linewidth=0.5)
    axis.set_xscale("log")
    axis.axhline(0.8, color=COLOR_TEXT, linestyle="--", linewidth=1)
    axis.axvline(threshold, color=COLOR_TEXT, linestyle=":", linewidth=1)
    axis.text(threshold, 0.98, f" high-income line\n US$ {threshold:,.0f}",
              transform=axis.get_xaxis_transform(), fontsize=8, va="top",
              color="#46505a")
    contradictions = data[data["c2_group"].isin(GROUPS)
                          & (data["c2_group"] != data["group"])]
    for _, row in contradictions.iterrows():
        axis.annotate(row["country"], (row["gni_atlas"], row["hdi"]),
                      textcoords="offset points", xytext=(5, -10), fontsize=8,
                      color=COLOR_ACCENT)
    axis.legend(loc="upper left")
    _style(axis, f"Figure 9 - HDI against GNI per capita, {base_year}",
           "GNI per capita, Atlas method (current US$, log scale)",
           "Human Development Index (0-1)")
    return _save(figure, path, "UNDP HDR; World Bank WDI NY.GNP.PCAP.CD and "
                 "income classification; labelled = C2 contradicts C1", dpi)


# --------------------------------------------------------------------------
# 10 cross-institutional agreement
# --------------------------------------------------------------------------

def fig10_agreement_matrix(sample: pd.DataFrame, base_year: int, path: Path,
                           dpi: int = 300) -> Path:
    data = sample.dropna(subset=["group"]).copy()
    data = data.sort_values(["c3_agreements", "group", "hdi"],
                            ascending=[False, True, False])
    columns = [("group", "C1 (HDI)"), ("imf_group", "IMF"),
               ("undp_group", "UNDP"), ("wb_group", "World Bank")]
    code = {"developed": 0, "developing": 1, "not classified": 2}
    matrix = np.array([[code.get(str(v), 2) for v in data[c]]
                       for c, _ in columns]).T
    cmap = ListedColormap([COLOR_DEVELOPED, COLOR_DEVELOPING, "#e3e7ec"])
    figure, axis = plt.subplots(figsize=(6.5, max(4, 0.19 * len(data) + 1.5)))
    axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                interpolation="nearest")
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels([label for _, label in columns])
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(data)))
    axis.set_yticklabels([f"{c}  ({int(a)}/3)" for c, a in
                          zip(data["country"], data["c3_agreements"])],
                         fontsize=7)
    axis.legend(handles=[Patch(color=COLOR_DEVELOPED, label="developed"),
                         Patch(color=COLOR_DEVELOPING, label="developing"),
                         Patch(color="#e3e7ec", label="not classified")],
                loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
                fontsize=8)
    axis.set_title(f"Figure 10 - Agreement with C1 at {base_year}",
                   loc="left", pad=28)
    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_visible(False)
    return _save(figure, path, "UNDP HDR (top-quartile rule), IMF WEO, World "
                 "Bank; (k/3) = institutions agreeing with C1", dpi)


# --------------------------------------------------------------------------
# 11 event qualification funnel
# --------------------------------------------------------------------------

def fig11_event_funnel(funnel: pd.DataFrame, path: Path,
                       dpi: int = 300) -> Path:
    world = funnel[funnel["population"] == "all GTD countries"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2),
                                         gridspec_kw={"width_ratios": [1.3,
                                                                       1]})
    positions = np.arange(len(world))
    left.bar(positions, world["events"], color=COLOR_NEUTRAL, width=0.62,
             zorder=2)
    for x, (_, row) in zip(positions, world.iterrows()):
        left.text(x, row["events"], f"{int(row['events']):,}\n"
                  f"{row['pct_of_window']:.0f}%", ha="center", va="bottom",
                  fontsize=8)
    left.set_xticks(positions)
    left.set_xticklabels([s for s in world["stage"]], fontsize=8)
    left.set_ylim(0, world["events"].max() * 1.18)
    _style(left, "Figure 11 - Event qualification funnel", "stage",
           "events (count), all GTD countries", grid_axis="y")

    styles = {"all GTD countries": (COLOR_NEUTRAL, "-", 2.2),
              "developed markets": (COLOR_DEVELOPED, "-", 1.8),
              "developing markets": (COLOR_DEVELOPING, "--", 1.8)}
    for population, (color, line, width) in styles.items():
        chunk = funnel[funnel["population"] == population]
        if chunk.empty:
            continue
        right.plot(np.arange(len(chunk)), chunk["pct_of_window"], line,
                   marker="o", color=color, linewidth=width, label=population)
    right.set_xticks(positions)
    right.set_xticklabels(list(world["stage"]), fontsize=8)
    right.set_ylim(0, 105)
    right.legend(loc="lower left")
    _style(right, "Share retained, by group", "stage",
           "% of events in window still qualifying", grid_axis="y")
    return _save(figure, path, GTD_SOURCE + "; counts in t5_2_event_funnel.csv",
                 dpi)


# --------------------------------------------------------------------------
# 12 composition of qualifying events
# --------------------------------------------------------------------------

def fig12_event_composition(events: pd.DataFrame, sample: pd.DataFrame,
                            path: Path, dpi: int = 300) -> Path:
    data = events.assign(group=events["iso3"].map(_group_of(sample)))
    data = data[data["group"].isin(GROUPS)]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    damaged = data[data["property"] == 1].copy()
    damaged["band"] = damaged["propextent"].fillna(4).astype(int)
    band_order = [(1, "catastrophic", "#5b1a26"), (2, "major", COLOR_ACCENT),
                  (3, "minor", "#e3a3ab"), (4, "unknown -> minor", "#f1d9dc")]
    for index, group in enumerate(GROUPS):
        chunk = damaged[damaged["group"] == group]
        total = max(len(chunk), 1)
        bottom = 0.0
        for band, label, color in band_order:
            share = (chunk["band"] == band).sum() / total * 100
            left.bar(index, share, bottom=bottom, color=color, width=0.55,
                     hatch="//" if band == 4 else None, edgecolor="white",
                     label=label if index == 0 else None, zorder=2)
            bottom += share
        left.text(index, 101, f"n = {len(chunk):,}", ha="center", fontsize=8)
    left.set_xticks(range(len(GROUPS)))
    left.set_xticklabels(GROUPS)
    left.set_ylim(0, 108)
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
                fontsize=8)
    _style(left, "Figure 12a - Extent of property damage", "",
           "% of qualifying events with property damage", grid_axis="y")

    casualty = (data["nkill"] >= 1) | (data["nwound"] >= 1)
    prop = data["property"] == 1
    kinds = pd.Series(np.select([prop & ~casualty, casualty & ~prop,
                                 casualty & prop],
                                ["property only", "casualties only", "both"],
                                default="other"), index=data.index)
    colors = {"property only": "#c9d4e0", "casualties only": COLOR_NEUTRAL,
              "both": COLOR_TEXT}
    for index, group in enumerate(GROUPS):
        chunk = kinds[data["group"] == group]
        total = max(len(chunk), 1)
        bottom = 0.0
        for kind, color in colors.items():
            share = (chunk == kind).sum() / total * 100
            right.bar(index, share, bottom=bottom, color=color, width=0.55,
                      label=kind if index == 0 else None, zorder=2)
            if share >= 6:
                right.text(index, bottom + share / 2, f"{share:.0f}%",
                           ha="center", va="center", fontsize=8,
                           color="white" if kind != "property only"
                           else COLOR_TEXT)
            bottom += share
        right.text(index, 101, f"n = {len(chunk):,}", ha="center", fontsize=8)
    right.set_xticks(range(len(GROUPS)))
    right.set_xticklabels(GROUPS)
    right.set_ylim(0, 108)
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3,
                 fontsize=8)
    _style(right, "Figure 12b - What qualified the event", "",
           "% of qualifying events", grid_axis="y")
    return _save(figure, path, GTD_SOURCE + "; hatched = propextent 4 recoded "
                 "to minor (File 3 §5.1)", dpi)


# --------------------------------------------------------------------------
# 13A and 14: why the log transform
# --------------------------------------------------------------------------

def fig13a_skewness_by_market(skew: pd.DataFrame, sample: pd.DataFrame,
                              path: Path, dpi: int = 300) -> Path:
    data = _labelled(skew, sample)
    undefined = int(data["skewness"].isna().sum())
    data = data.dropna(subset=["skewness"]).sort_values("skewness")
    figure, axis = plt.subplots(figsize=(8.5, max(4, 0.19 * len(data) + 1.5)))
    axis.barh(data["country"], data["skewness"],
              color=data["group"].map(GROUP_COLORS), height=0.75, zorder=2)
    axis.axvline(0, color=COLOR_TEXT, linewidth=0.8)
    axis.tick_params(axis="y", labelsize=7)
    _group_legend(axis, "lower right")
    _style(axis, "Figure 13A - Skewness of annual fatality counts (raw NK), "
                 "by market", "skewness of the market's own annual NK series,"
                              " untransformed", "", grid_axis="x")
    note = (f"; {undefined} markets with a constant NK series (skewness "
            f"undefined) omitted" if undefined else "")
    return _save(figure, path, GTD_SOURCE + note, dpi)


def fig14_transformation(tsi_table: pd.DataFrame, path: Path,
                         dpi: int = 300) -> Path:
    stages = [("NK", "raw NK (fatalities per market-year)"),
              ("ln_NK", "ln(1 + NK)"),
              ("z_NK", "z(ln(1 + NK)), pooled")]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (column, label) in zip(axes, stages):
        values = tsi_table[column].dropna()
        axis.hist(values, bins=40, color=COLOR_NEUTRAL, zorder=2)
        axis.text(0.97, 0.95, f"skewness {values.skew():.2f}\n"
                  f"max {((values.max() - values.mean()) / values.std()):.1f} "
                  f"SD above mean", transform=axis.transAxes, ha="right",
                  va="top", fontsize=8.5,
                  bbox=dict(facecolor="white", edgecolor=COLOR_GRID))
        _style(axis, "", label, "market-years (count)", grid_axis="y")
    axes[0].set_title("Figure 14 - Why the transformation is necessary",
                      loc="left", pad=10)
    return _save(figure, path, GTD_SOURCE + "; the z-score keeps the log's "
                 "shape - standardising is linear", dpi)


# --------------------------------------------------------------------------
# 15 TSI heatmap
# --------------------------------------------------------------------------

def fig15_tsi_heatmap(tsi_table: pd.DataFrame, sample: pd.DataFrame,
                      path: Path, dpi: int = 300) -> Path:
    data = _labelled(tsi_table, sample)
    order = (data.groupby(["group", "country"])["TSI"].mean().reset_index()
             .assign(g=lambda t: t["group"].map({"developed": 0,
                                                 "developing": 1}))
             .sort_values(["g", "TSI"], ascending=[True, False]))
    matrix = (data.pivot(index="country", columns="year", values="TSI")
              .reindex(order["country"]))
    figure, axis = plt.subplots(figsize=(11, max(4, 0.19 * len(matrix) + 2)))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="YlOrRd",
                        interpolation="nearest")
    axis.set_yticks(range(len(matrix)))
    axis.set_yticklabels(matrix.index, fontsize=7)
    _colour_ticklabels(axis, matrix.index, order["group"])
    years = list(matrix.columns)
    axis.set_xticks(range(len(years)))
    axis.set_xticklabels(years, rotation=90, fontsize=8)
    n_developed = int((order["group"] == "developed").sum())
    if 0 < n_developed < len(order):
        axis.axhline(n_developed - 0.5, color=COLOR_TEXT, linewidth=1.5)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label("TSI (standard deviations from the pooled mean)")
    axis.set_title("Figure 15 - Terrorism Severity Index, market x year "
                   "(developed above the line)", loc="left", pad=10)
    return _save(figure, path, GTD_SOURCE + "; TSI of File 3 §7", dpi)


# --------------------------------------------------------------------------
# 16 correlation matrix
# --------------------------------------------------------------------------

def fig16_correlation_matrix(corr: pd.DataFrame, labels: Dict[str, str],
                             path: Path, dpi: int = 300) -> Path:
    names = [labels.get(c, c) for c in corr.columns]
    values = corr.to_numpy().copy()
    mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    values[mask] = np.nan
    size = max(6, 0.62 * len(names) + 2)
    figure, axis = plt.subplots(figsize=(size, size * 0.85))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(names)):
        for j in range(i + 1):
            if np.isfinite(values[i, j]):
                axis.text(j, i, f"{values[i, j]:.2f}", ha="center",
                          va="center", fontsize=7,
                          color="white" if abs(values[i, j]) > 0.6
                          else COLOR_TEXT)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=8)
    figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02,
                    label="Pearson correlation")
    axis.set_title("Figure 16 - Correlation matrix of the model variables",
                   loc="left", pad=10)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    return _save(figure, path, "estimation panel (pairwise complete); "
                 f"{WDI_SOURCE}, UCDP, GTD", dpi)


# --------------------------------------------------------------------------
# 17, 18: the treatment variable by group
# --------------------------------------------------------------------------

def fig17_components_over_time(tsi_table: pd.DataFrame, sample: pd.DataFrame,
                               path: Path, dpi: int = 300) -> Path:
    data = tsi_table.assign(group=tsi_table["iso3"].map(_group_of(sample)))
    figure, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharex=True,
                                sharey=True)
    for row, group in enumerate(GROUPS):
        chunk = data[data["group"] == group]
        for col, component in enumerate(COMPONENTS):
            axis = axes[row, col]
            yearly = chunk.groupby("year")[f"z_{component}"].mean()
            axis.plot(yearly.index, yearly.values, color=GROUP_COLORS[group],
                      linewidth=2)
            axis.axhline(0, color=COLOR_TEXT, linewidth=0.6)
            _style(axis, f"{component} - {group}",
                   "year" if row == 1 else "",
                   "mean z(ln(1+x)) (SD units)" if col == 0 else "")
            axis.title.set_fontsize(10)
    figure.suptitle("Figure 17 - TSI components over time, by group "
                    "(shared scale)", x=0.01, ha="left", fontweight="bold")
    return _save(figure, path, GTD_SOURCE + "; NA attacks, NK killed, NI "
                 "injured, PD property score", dpi)


def fig18_tsi_distribution(tsi_table: pd.DataFrame, sample: pd.DataFrame,
                           path: Path, dpi: int = 300) -> Path:
    data = tsi_table.assign(group=tsi_table["iso3"].map(_group_of(sample)))
    bins = np.linspace(data["TSI"].min(), data["TSI"].max(), 36)
    figure, axis = plt.subplots(figsize=(9.5, 5))
    floor = data["TSI"].min()
    for group in GROUPS:
        values = data.loc[data["group"] == group, "TSI"].dropna()
        at_floor = (np.isclose(values, floor)).mean() * 100
        axis.hist(values, bins=bins, density=True, histtype="stepfilled",
                  alpha=0.35, color=GROUP_COLORS[group], zorder=2)
        axis.hist(values, bins=bins, density=True, histtype="step",
                  linewidth=1.8, color=GROUP_COLORS[group], zorder=3,
                  label=f"{group}: mean {values.mean():+.2f}, "
                        f"{at_floor:.0f}% with no qualifying event")
    axis.legend(loc="upper right")
    _style(axis, "Figure 18 - Distribution of the TSI by group",
           "TSI (standard deviations from the pooled mean)", "density")
    return _save(figure, path, GTD_SOURCE, dpi)


# --------------------------------------------------------------------------
# 19 outcomes over time
# --------------------------------------------------------------------------

def fig19_outcomes_over_time(data: pd.DataFrame, outcomes: Dict[str, object],
                             path: Path, dpi: int = 300) -> Path:
    figure, axes = plt.subplots(len(outcomes), 1, figsize=(10, 4 *
                                                           len(outcomes)),
                                sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (name, spec) in zip(axes, outcomes.items()):
        for start, end in ((2008, 2009), (2020, 2020)):
            axis.axvspan(start - 0.5, end + 0.5, color="#eceff3", zorder=0)
        for group in GROUPS:
            chunk = data[data["group"] == group].groupby("year")[name]
            mean, q25, q75 = chunk.mean(), chunk.quantile(0.25), \
                chunk.quantile(0.75)
            axis.fill_between(mean.index, q25, q75, alpha=0.18,
                              color=GROUP_COLORS[group], zorder=1)
            axis.plot(mean.index, mean.values, "-o", markersize=3,
                      color=GROUP_COLORS[group], linewidth=2,
                      label=f"{group}: mean, band = interquartile range")
        axis.axhline(0, color=COLOR_TEXT, linewidth=0.6)
        axis.legend(loc="best", fontsize=8)
        _style(axis, f"{spec.label} ({spec.hypothesis})",
               "year" if axis is axes[-1] else "", spec.unit)
    figure.suptitle("Figure 19 - Outcome variables over time (shaded: "
                    "2008-09 and 2020)", x=0.01, ha="left",
                    fontweight="bold")
    return _save(figure, path, f"{WDI_SOURCE} NY.GDP.MKTP.KD.ZG, "
                 f"BX/BM.KLT.DINV.WD.GD.ZS; sample markets", dpi)


# --------------------------------------------------------------------------
# 20-24: results
# --------------------------------------------------------------------------

def fig20_21_coefplot(fits: list, label: str, unit: str, number: int,
                      path: Path, dpi: int = 300) -> Path:
    rows = []
    for fit in fits:
        row = fit.get("TSI")
        if row is not None:
            rows.append((fit.title, row["coef"], row["ci_low"],
                         row["ci_high"], row["p"]))
    figure, axis = plt.subplots(figsize=(9, 1.3 + 0.8 * len(rows)))
    for index, (title, coef, low, high, p) in enumerate(rows):
        color = COLOR_ACCENT if p < 0.05 else COLOR_NEUTRAL
        axis.errorbar(coef, index, xerr=[[coef - low], [high - coef]],
                      fmt="o", color=color, capsize=4, markersize=7)
        axis.annotate(f"{coef:+.3f}", (coef, index), textcoords="offset "
                      "points", xytext=(0, 9), ha="center", fontsize=8,
                      color=color)
    axis.axvline(0, color=COLOR_TEXT, linestyle="--", linewidth=1)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([r[0] for r in rows])
    axis.invert_yaxis()
    axis.margins(y=0.3)
    _style(axis, f"Figure {number} - Coefficient on TSI(t), {label}",
           f"change in {label} ({unit}) per 1 SD of TSI, 95% CI "
           f"(red = p < 0.05)", "")
    return _save(figure, path, "regressions of §8.4/§8.5, all countries; "
                 "country FE in every column", dpi)


def fig22_marginal_effects(effects: pd.DataFrame, labels: Dict[str, str],
                           path: Path, dpi: int = 300) -> Path:
    data = effects[(effects["horizon"] == "t")
                   & effects["group"].isin(GROUPS)].reset_index(drop=True)
    difference = effects[(effects["horizon"] == "t")
                         & effects["group"].str.startswith("difference")]
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    ticks = []
    for index, row in data.iterrows():
        axis.errorbar(row["coef"], index,
                      xerr=[[row["coef"] - row["ci_low"]],
                            [row["ci_high"] - row["coef"]]],
                      fmt="o", color=GROUP_COLORS[row["group"]], capsize=4,
                      markersize=8)
        axis.annotate(f"{row['coef']:+.3f}{row['stars']}",
                      (row["coef"], index), textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=8,
                      color=GROUP_COLORS[row["group"]])
        ticks.append(f"{labels.get(row['outcome'], row['outcome'])}\n"
                     f"{row['group']}")
    axis.axvline(0, color=COLOR_TEXT, linestyle="--", linewidth=1)
    axis.set_yticks(range(len(ticks)))
    axis.set_yticklabels(ticks, fontsize=9)
    axis.invert_yaxis()
    axis.margins(y=0.2)
    notes = "; ".join(f"{labels.get(r['outcome'], r['outcome'])}: b3 = "
                      f"{r['coef']:+.3f} (p = {r['p']:.3f})"
                      for _, r in difference.iterrows())
    _style(axis, "Figure 22 - Effect of a 1 SD rise in the TSI, by group",
           "change in the outcome (percentage points), 95% CI", "")
    return _save(figure, path, f"regression 4 (§7.3b): developing = b1, "
                 f"developed = b1 + b3. {notes}", dpi)


def _residualise(data: pd.DataFrame, column: str,
                 others: Sequence[str]) -> np.ndarray:
    formula = f"{column} ~ " + " + ".join(list(others) + ["C(iso3)",
                                                          "C(year)"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(smf.ols(formula, data=data).fit().resid)


def fig23_binned_scatter(samples: Dict[str, pd.DataFrame],
                         outcomes: Dict[str, object], controls: List[str],
                         path: Path, bins: int = 15,
                         dpi: int = 300) -> Path:
    """Within each group, the outcome and TSI(t) residualised on TSI(t-1),
    the controls and both fixed effects (Frisch-Waugh), so the fitted line's
    slope is that group's split-sample coefficient (regressions 2 and 3)."""
    figure, axes = plt.subplots(1, len(outcomes), figsize=(6.2 *
                                                          len(outcomes), 5))
    axes = np.atleast_1d(axes)
    for axis, (name, spec) in zip(axes, outcomes.items()):
        data = samples[name]
        for group in GROUPS:
            rows = data[data["group"] == group].reset_index(drop=True)
            if rows["iso3"].nunique() < 2:
                continue
            others = ["TSI_lag1"] + [c for c in controls
                                     if rows[c].std(ddof=0) > 0]
            x = _residualise(rows, "TSI", others)
            y = _residualise(rows, name, others)
            edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
            which = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
            means = pd.DataFrame({"x": x, "y": y, "bin": which}).groupby(
                "bin").mean()
            axis.scatter(means["x"], means["y"], color=GROUP_COLORS[group],
                         s=34, zorder=3)
            slope = np.polyfit(x, y, 1)
            line = np.linspace(x.min(), x.max(), 50)
            axis.plot(line, np.polyval(slope, line),
                      color=GROUP_COLORS[group], linewidth=2,
                      label=f"{group}: slope {slope[0]:+.3f}")
        axis.axhline(0, color="#9aa4ae", linewidth=0.8)
        axis.axvline(0, color="#9aa4ae", linewidth=0.8)
        axis.legend(loc="best", fontsize=8)
        _style(axis, f"{spec.label}",
               "TSI(t), residualised (SD units)",
               f"{spec.label}, residualised ({spec.unit})")
    figure.suptitle(f"Figure 23 - Binned scatter ({bins} quantile bins), "
                    f"TSI against outcome, by group", x=0.01, ha="left",
                    fontweight="bold")
    return _save(figure, path, "estimation panel; residuals on TSI(t-1), "
                 "controls, market and year FE within each group", dpi)


def fig24_component_coefficients(components: pd.DataFrame,
                                 labels: Dict[str, str], path: Path,
                                 dpi: int = 300) -> Path:
    data = components[(components["horizon"] == "t")
                      & components["sample"].isin(GROUPS)]
    outcomes = list(dict.fromkeys(data["outcome"]))
    figure, axes = plt.subplots(1, len(outcomes), figsize=(6.2 *
                                                          len(outcomes), 4.8))
    axes = np.atleast_1d(axes)
    width = 0.36
    for axis, outcome in zip(axes, outcomes):
        chunk = data[data["outcome"] == outcome]
        for offset, group in zip((-width / 2, width / 2), GROUPS):
            rows = (chunk[chunk["sample"] == group].set_index("component")
                    .reindex(COMPONENTS))
            x = np.arange(len(COMPONENTS)) + offset
            axis.bar(x, rows["coef"], width=width, color=GROUP_COLORS[group],
                     label=group, zorder=2)
            axis.errorbar(x, rows["coef"],
                          yerr=[rows["coef"] - rows["ci_low"],
                                rows["ci_high"] - rows["coef"]],
                          fmt="none", ecolor=COLOR_TEXT, capsize=3, zorder=3)
        axis.axhline(0, color=COLOR_TEXT, linewidth=0.8)
        axis.set_xticks(range(len(COMPONENTS)))
        axis.set_xticklabels([f"{c}\n{COMPONENT_LABELS[c]}"
                              for c in COMPONENTS], fontsize=8)
        axis.legend(loc="best", fontsize=8)
        _style(axis, labels.get(outcome, outcome), "",
               "coefficient per 1 SD of the component (95% CI)",
               grid_axis="y")
    figure.suptitle("Figure 24 - Component coefficients, by group", x=0.01,
                    ha="left", fontweight="bold")
    return _save(figure, path, "§8.7 decomposition: z(ln(1+x)) of each "
                 "component in place of the TSI, split samples", dpi)
