# -*- coding: utf-8 -*-
"""STEP 3 - the analysis.

For every metric declared in config.py, independently:

    A  before terrorism   metric ~ controls + country FE + year FE
    B  after terrorism    metric ~ controls + terror + country FE + year FE
    C  heterogeneity      B + terror x developed
    D  long run           metric ~ controls + terror(t ... t-k) + FE

and then writes the tables into outputs/tables/, the figures into
outputs/figures/, and one readable page at outputs/report.html.

Run it from PyCharm with the green arrow. Two switches, if you want them:

    --sample   run on the synthetic data in data/sample/ instead, where the
               true effect is known, to prove the machinery works
    --metric X analyse only metric X
"""

import argparse
import sys
import webbrowser

import pandas as pd

import config
from seminar import (countries, panel, paths, pipeline, regression, report,
                     sample_data, terror, viz)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true",
                        help="use the synthetic data with a known planted "
                             "effect instead of data/metrics")
    parser.add_argument("--metric", default=None,
                        help="analyse only this metric")
    parser.add_argument("--open", action="store_true",
                        help="open the HTML report when finished")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()
    first_year, last_year = config.PERIOD

    print("=" * 70)
    print(f"STEP 3  analysis     period {first_year}-{last_year}     "
          f"terrorism measure: {config.TERROR_MEASURE}")
    print("=" * 70)

    if args.sample:
        print("\nSAMPLE MODE - synthetic data with a known planted effect\n")
        sample_data.generate(config.PERIOD)
        metric_specs = dict(sample_data.SAMPLE_METRICS)
        control_specs = dict(sample_data.SAMPLE_CONTROLS)
        filter_specs = dict(sample_data.SAMPLE_FILTERS)
        active_filter = "gni_per_capita"
        metrics_dir = sample_data.SAMPLE_METRICS_DIR
        controls_dir = sample_data.SAMPLE_CONTROLS_DIR
        filters_dir = sample_data.SAMPLE_FILTERS_DIR
        terror_country_year = sample_data.load_sample_terror()
        terror_measure = "log_killed"
    else:
        metric_specs = dict(config.METRICS)
        control_specs = dict(config.CONTROLS)
        filter_specs = dict(config.FILTERS)
        active_filter = config.ACTIVE_FILTER
        metrics_dir = paths.METRICS_DIR
        controls_dir = paths.CONTROLS_DIR
        filters_dir = paths.FILTERS_DIR
        terror_measure = config.TERROR_MEASURE
        if not paths.GTD_COUNTRY_YEAR_PARQUET.exists():
            print("\nERROR: no terrorism cache. Run 1_build_terror_cache.py "
                  "first (once, it takes a few minutes).")
            return 1
        terror_country_year = pd.read_parquet(paths.GTD_COUNTRY_YEAR_PARQUET)

    if args.metric:
        if args.metric not in metric_specs:
            print(f"ERROR: '{args.metric}' is not declared. Available: "
                  f"{sorted(metric_specs)}")
            return 1
        metric_specs = {args.metric: metric_specs[args.metric]}

    data = pipeline.load_all(
        metric_specs, control_specs, filter_specs, active_filter, config.PERIOD,
        config.COUNTRY_SUBSET, metrics_dir, controls_dir, filters_dir,
    )
    pipeline.report_problems(data.problems)
    if data.problems:
        print("\nFix the problems above (2_check_data.py explains them in "
              "more detail), then run this again.")
        return 1

    print("\nBuilding one panel per metric")
    panels = {}
    for name, table in data.metrics.items():
        built = panel.build_panel(
            metric_name=name, metric_spec=metric_specs[name],
            metric_table=table, terror_country_year=terror_country_year,
            controls=data.controls, control_specs=control_specs,
            classification=data.classification, period=config.PERIOD,
            terror_measure=terror_measure, lags=config.LAGS,
            population=data.population_raw,
        )
        panels[name] = built
        print(f"  {built.summary()}")
        for note in built.notes:
            print(f"      note: {note}")

    print("\nEstimating")
    results = regression.run_all(panels, terror_measure, config.LAGS)

    print("\nWriting tables and figures")
    figures = {}
    for name, result in results.items():
        report.write_tables(result)
        if not result.ok:
            continue
        built = panels[name]
        metric_figures = {}
        metric_figures["before_after"] = viz.before_after_chart(
            result, paths.FIGURES_DIR / f"{name}_before_after.png",
            config.FIGURE_DPI)
        metric_figures["forest"] = viz.effect_forest(
            result, paths.FIGURES_DIR / f"{name}_effect.png", config.FIGURE_DPI)
        metric_figures["lags"] = viz.lag_profile_chart(
            result, paths.FIGURES_DIR / f"{name}_lags.png", config.FIGURE_DPI)
        try:
            frame = regression.partial_regression_frame(built)
            metric_figures["partial"] = viz.partial_regression_plot(
                frame, result, paths.FIGURES_DIR / f"{name}_partial.png",
                config.FIGURE_DPI)
        except Exception as error:                       # noqa: BLE001
            print(f"  {name}: partial regression plot skipped ({error})")
        metric_figures["trend"] = viz.group_trend_chart(
            built, result, paths.FIGURES_DIR / f"{name}_trend.png",
            config.FIGURE_DPI)
        matrix = panel.coverage_matrix(data.metrics[name], name, config.PERIOD)
        metric_figures["coverage"] = viz.coverage_heatmap(
            matrix, name, paths.FIGURES_DIR / f"coverage_{name}.png",
            config.MAX_COUNTRIES_IN_HEATMAP, config.FIGURE_DPI)
        figures[name] = {k: v for k, v in metric_figures.items() if v}
        print(f"  {name}: {len(figures[name])} figures")

    figures["_overview"] = {}
    cross = viz.cross_metric_summary(
        results, paths.FIGURES_DIR / "all_metrics.png", config.FIGURE_DPI)
    if cross:
        figures["_overview"]["cross_metric"] = cross
    in_period = terror_country_year[
        terror_country_year["year"].between(first_year, last_year)]
    overview = viz.terror_overview(
        in_period, paths.FIGURES_DIR / "terrorism_over_time.png",
        config.FIGURE_DPI)
    if overview:
        figures["_overview"]["terror_overview"] = overview

    summary_path = report.write_summary_table(results)
    print(f"  {summary_path}")

    run_info = {
        "period": f"{first_year}-{last_year}",
        "terrorism measure": terror_measure,
        "lags": str(config.LAGS),
        "classification": f"{active_filter} ({data.filter_spec.mode})",
        "controls": ", ".join(data.controls) or "none",
        "metrics analysed": f"{sum(1 for r in results.values() if r.ok)} of "
                            f"{len(results)}",
        "GTD filters": f"all criteria={config.GTD_REQUIRE_ALL_CRITERIA}, "
                       f"drop doubted={config.GTD_DROP_DOUBTED}",
        "mode": "SAMPLE (synthetic data)" if args.sample else "real data",
    }
    report_path = report.build_report(results, figures, run_info)
    print(f"  {report_path}")

    if args.sample:
        print("\n" + "=" * 70)
        print("DID THE PIPELINE RECOVER THE PLANTED EFFECT?")
        print("=" * 70)
        for name, result in results.items():
            if result.ok:
                print(f"\n{name}")
                print(sample_data.check_recovery(result))

    countries.write_unmatched_report()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, result in results.items():
        if not result.ok:
            print(f"{name:<22} skipped - {result.skipped_reason}")
            continue
        comparison = result.comparison
        print(f"{name:<22} terror {comparison['terror_coef']:+.4g} "
              f"(p={comparison['terror_p']:.3f})   within-R2 "
              f"{comparison['r2_within_before']:.3f} -> "
              f"{comparison['r2_within_after']:.3f}")
        for line in result.interpretation[:2]:
            print(f"    {line}")

    print(f"\nFull report: {paths.REPORT_HTML}")
    if args.open:
        webbrowser.open(paths.REPORT_HTML.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
