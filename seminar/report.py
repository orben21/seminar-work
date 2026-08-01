"""Write the results out: tidy CSV tables plus one self-contained HTML report.

The CSVs are what you cite in the paper. The HTML report is one page holding
every figure and table, so you can read the whole result in one scroll and
screenshot what you need.
"""

from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import paths
from .regression import MetricResult

CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; background: #f4f6f8; color: #22282e; line-height: 1.5; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 32px 24px 80px; }
h1 { font-size: 28px; margin: 0 0 4px; }
h2 { font-size: 21px; margin: 40px 0 4px; padding-top: 20px;
     border-top: 2px solid #d7dde3; }
h3 { font-size: 15px; margin: 26px 0 8px; color: #46505a;
     text-transform: uppercase; letter-spacing: .04em; }
.sub { color: #667180; margin: 0 0 24px; }
.card { background: #fff; border: 1px solid #e2e7ec; border-radius: 8px;
        padding: 20px 22px; margin: 16px 0; }
.headline { background: #1f4e79; color: #fff; border-radius: 8px;
            padding: 18px 22px; margin: 16px 0; }
.headline b { font-size: 17px; }
table { border-collapse: collapse; width: 100%; font-size: 13px;
        margin: 8px 0 4px; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid #e8ecf0; }
th:first-child, td:first-child { text-align: left; }
th { background: #eef2f6; font-weight: 600; }
tr:last-child td { border-bottom: none; }
img { max-width: 100%; height: auto; display: block; margin: 8px 0;
      border: 1px solid #e2e7ec; border-radius: 6px; }
.note { background: #fff8e6; border-left: 4px solid #e0a800; padding: 10px 14px;
        margin: 10px 0; font-size: 13px; }
.skip { background: #fdecea; border-left: 4px solid #c0392b; padding: 10px 14px;
        font-size: 13px; }
.mono { font-family: Consolas, monospace; font-size: 12px; color: #46505a; }
ul.plain { margin: 6px 0; padding-left: 20px; }
.toc a { display: inline-block; margin-right: 14px; color: #1f4e79; }
"""


# --------------------------------------------------------------------------
# CSV output
# --------------------------------------------------------------------------

def write_tables(result: MetricResult,
                 directory: Path = paths.TABLES_DIR) -> List[Path]:
    """One folder per metric, holding every number behind the figures."""
    folder = Path(directory) / result.metric
    folder.mkdir(parents=True, exist_ok=True)
    written = []

    if not result.ok:
        note = pd.DataFrame([{"metric": result.metric,
                              "skipped_reason": result.skipped_reason}])
        path = folder / "skipped.csv"
        note.to_csv(path, index=False, encoding="utf-8-sig")
        return [path]

    frames = []
    for key, model in result.models.items():
        table = model.coefficients.copy()
        table.insert(0, "model", f"{key} - {model.name}")
        frames.append(table)
    if frames:
        path = folder / "coefficients.csv"
        pd.concat(frames, ignore_index=True).to_csv(
            path, index=False, encoding="utf-8-sig")
        written.append(path)

    rows = []
    for key, model in result.models.items():
        rows.append({
            "model": key, "name": model.name, "formula": model.formula,
            "n_obs": model.n_obs, "n_countries": model.n_countries,
            "r2": model.r2, "r2_within": model.r2_within, "rss": model.rss,
        })
    path = folder / "model_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    written.append(path)

    path = folder / "terror_effect.csv"
    pd.DataFrame([result.comparison]).to_csv(path, index=False,
                                             encoding="utf-8-sig")
    written.append(path)

    for name, table in (("control_shifts", result.control_shifts),
                        ("group_effects", result.group_effects),
                        ("lag_profile", result.lag_profile),
                        ("vif", result.vif)):
        if isinstance(table, pd.DataFrame) and not table.empty:
            path = folder / f"{name}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            written.append(path)

    if result.long_run:
        path = folder / "long_run.csv"
        pd.DataFrame([result.long_run]).to_csv(path, index=False,
                                               encoding="utf-8-sig")
        written.append(path)

    return written


def write_summary_table(results: Dict[str, MetricResult],
                        directory: Path = paths.TABLES_DIR) -> Path:
    """One row per metric: the paper's summary table."""
    rows = []
    for name, result in results.items():
        if not result.ok:
            rows.append({"metric": name, "status": "skipped",
                         "reason": result.skipped_reason})
            continue
        comparison = result.comparison
        row = {
            "metric": name,
            "label": result.label,
            "status": "ok",
            "n_obs": result.n_obs,
            "n_countries": result.n_countries,
            "years": f"{result.year_min}-{result.year_max}",
            "terror_measure": result.terror_measure,
            "r2_within_before": comparison.get("r2_within_before"),
            "r2_within_after": comparison.get("r2_within_after"),
            "r2_within_gain": comparison.get("r2_within_gain"),
            "terror_coef": comparison.get("terror_coef"),
            "terror_se": comparison.get("terror_se"),
            "terror_p": comparison.get("terror_p"),
        }
        if not result.group_effects.empty:
            table = result.group_effects.set_index("group")
            for group in ("developing", "developed"):
                if group in table.index:
                    row[f"coef_{group}"] = table.loc[group, "coef"]
                    row[f"p_{group}"] = table.loc[group, "p"]
            key = "difference (developed - developing)"
            if key in table.index:
                row["coef_difference"] = table.loc[key, "coef"]
                row["p_difference"] = table.loc[key, "p"]
        if result.long_run:
            row["long_run_cumulative"] = result.long_run.get("cumulative")
            row["long_run_p"] = result.long_run.get("p")
        rows.append(row)

    path = Path(directory) / "summary_all_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------

def _embed(path: Optional[Path]) -> str:
    """Inline a PNG so the report is one self-contained file."""
    if path is None or not Path(path).exists():
        return ""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{Path(path).name}">'


def _table_html(table: pd.DataFrame, float_format: str = "{:.4g}") -> str:
    if not isinstance(table, pd.DataFrame) or table.empty:
        return ""
    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda v: "" if pd.isna(v) else float_format.format(v))
    return display.to_html(index=False, border=0, escape=False)


def _fmt(value, spec="{:.4g}", fallback="n/a"):
    try:
        if value is None or not np.isfinite(value):
            return fallback
        return spec.format(value)
    except (TypeError, ValueError):
        return fallback


def build_report(results: Dict[str, MetricResult],
                 figures: Dict[str, Dict[str, Path]],
                 run_info: Dict[str, str],
                 path: Path = paths.REPORT_HTML) -> Path:
    """Assemble everything into outputs/report.html."""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Terrorism and country metrics</title>",
        f"<style>{CSS}</style></head><body><div class='wrap'>",
        "<h1>The effect of terrorism on long-term country metrics</h1>",
        f"<p class='sub'>Developed vs developing countries &middot; "
        f"generated {now}</p>",
    ]

    parts.append("<div class='card'><h3>This run</h3><table>")
    for key, value in run_info.items():
        parts.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    parts.append("</table></div>")

    usable = {k: v for k, v in results.items() if v.ok}
    links = " ".join(f"<a href='#{name}'>{result.label or name}</a>"
                     for name, result in results.items())
    parts.append(f"<p class='toc'>{links}</p>")

    overview = figures.get("_overview", {})
    if overview.get("cross_metric"):
        parts.append("<h2>All metrics at a glance</h2>")
        parts.append("<div class='card'>")
        parts.append(_embed(overview["cross_metric"]))
        parts.append("<p class='mono'>Each estimate is the change in the "
                     "metric, in its own standard deviations, per one standard "
                     "deviation of terrorism. That makes metrics measured in "
                     "different units directly comparable.</p></div>")
    if overview.get("terror_overview"):
        parts.append(f"<div class='card'>{_embed(overview['terror_overview'])}"
                     f"</div>")

    for name, result in results.items():
        parts.append(f"<h2 id='{name}'>{result.label or name}</h2>")
        if not result.ok:
            parts.append(f"<div class='skip'><b>Skipped.</b> "
                         f"{result.skipped_reason}</div>")
            continue

        comparison = result.comparison
        headline = (
            f"<b>Adding terrorism changed the within-country R&sup2; from "
            f"{_fmt(comparison.get('r2_within_before'), '{:.4f}')} to "
            f"{_fmt(comparison.get('r2_within_after'), '{:.4f}')}.</b><br>"
            f"Terrorism coefficient "
            f"{_fmt(comparison.get('terror_coef'), '{:+.4g}')} "
            f"(p = {_fmt(comparison.get('terror_p'), '{:.4f}')}), "
            f"on {result.n_obs:,} country-years from {result.n_countries} "
            f"countries, {result.year_min}-{result.year_max}."
        )
        parts.append(f"<div class='headline'>{headline}</div>")

        if result.interpretation:
            items = "".join(f"<li>{line}</li>" for line in result.interpretation)
            parts.append(f"<div class='card'><h3>What this means</h3>"
                         f"<ul class='plain'>{items}</ul></div>")

        metric_figures = figures.get(name, {})
        for key, caption in (
            ("before_after", "Before and after adding terrorism"),
            ("forest", "The effect, developed vs developing"),
            ("lags", "How long the effect lasts"),
            ("partial", "The isolated effect of terrorism"),
            ("trend", "The metric and terrorism over time"),
            ("coverage", "Data coverage"),
        ):
            figure_path = metric_figures.get(key)
            if figure_path:
                parts.append(f"<div class='card'><h3>{caption}</h3>"
                             f"{_embed(figure_path)}</div>")

        parts.append("<div class='card'><h3>Model comparison</h3>")
        rows = [{"model": f"{key} - {model.name}", "R2": model.r2,
                 "within R2": model.r2_within, "obs": model.n_obs,
                 "formula": model.formula}
                for key, model in result.models.items()]
        parts.append(_table_html(pd.DataFrame(rows)))
        parts.append("</div>")

        parts.append("<div class='card'><h3>Coefficients</h3>")
        for key, model in result.models.items():
            parts.append(f"<p class='mono'>{key} - {model.name}</p>")
            parts.append(_table_html(model.coefficients))
        parts.append("</div>")

        for title, table in (("How the controls moved", result.control_shifts),
                             ("Effect by group", result.group_effects),
                             ("Effect by lag", result.lag_profile),
                             ("Multicollinearity (VIF)", result.vif)):
            if isinstance(table, pd.DataFrame) and not table.empty:
                parts.append(f"<div class='card'><h3>{title}</h3>"
                             f"{_table_html(table)}</div>")

        if result.notes:
            items = "".join(f"<li>{note}</li>" for note in result.notes)
            parts.append(f"<div class='note'><b>Notes on this sample</b>"
                         f"<ul class='plain'>{items}</ul></div>")

    parts.append(
        "<h2>How to read this</h2><div class='card'>"
        "<p>Every model includes country fixed effects and year fixed effects. "
        "Country effects absorb everything permanent about a country - its "
        "institutions, geography, and long-run level of the metric. Year "
        "effects absorb whatever hit the whole world in a given year, such as "
        "the 2008 financial crisis. What is left for the terrorism coefficient "
        "is the within-country, relative-to-the-world variation: that is what "
        "isolating the effect of terrorism means here.</p>"
        "<p>Standard errors are clustered by country, because a country's "
        "shocks persist across years.</p>"
        "<p><b>Limitation to state in the paper:</b> terrorism is not randomly "
        "assigned. Weak economies may attract terrorism as much as terrorism "
        "weakens economies. Fixed effects and lagged terrorism reduce this "
        "concern but do not eliminate it, so these estimates are associations "
        "under strong controls, not proven causation.</p>"
        "<p>The GTD has no records at all for 1993, so that year is excluded "
        "rather than counted as a year without terrorism.</p>"
        "</div>")

    parts.append("</div></body></html>")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
