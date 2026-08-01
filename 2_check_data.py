# -*- coding: utf-8 -*-
"""STEP 2 - check that every declaration in config.py matches the real files.

Run this every time you add a data file or change config.py. Nothing is
estimated here; it only reports what you have and what is wrong, so problems
surface as a plain sentence instead of a stack trace halfway through a run.

It prints, for every declared file: how many rows, how many countries, which
years, how much is missing - and writes a coverage heatmap per metric into
outputs/figures/.
"""

import sys

import pandas as pd

import config
from seminar import countries, panel, paths, pipeline, terror, viz


def main() -> int:
    paths.ensure_dirs()
    first_year, last_year = config.PERIOD

    print("=" * 70)
    print(f"STEP 2  checking your data     period {first_year}-{last_year}")
    print("=" * 70)

    data = pipeline.load_all(
        config.METRICS, config.CONTROLS, config.FILTERS, config.ACTIVE_FILTER,
        config.PERIOD, config.COUNTRY_SUBSET,
    )

    print("\nDeclared filter files")
    for name, spec in config.FILTERS.items():
        path = paths.FILTERS_DIR / spec.file
        active = name == config.ACTIVE_FILTER
        mark = " (ACTIVE)" if active else ""
        if path.exists():
            print(f"  {name:<22} found{mark}")
        elif active:
            data.problems.append(
                f"ACTIVE_FILTER='{name}' points at {path}, which does not "
                f"exist.")
        else:
            print(f"  {name:<22} MISSING: {spec.file} is declared but not in "
                  f"data/filters/{mark}")
            print(f"  {'':<22} fine for now - it is not the active filter - "
                  f"but ACTIVE_FILTER='{name}' would fail until you add it")

    print("\nTerrorism data")
    country_year = None
    if not paths.GTD_COUNTRY_YEAR_PARQUET.exists():
        data.problems.append(
            "no terrorism cache yet. Run 1_build_terror_cache.py first.")
        print("  MISSING - run 1_build_terror_cache.py")
    else:
        country_year = pd.read_parquet(paths.GTD_COUNTRY_YEAR_PARQUET)
        in_period = country_year[
            country_year["year"].between(first_year, last_year)]
        print(terror.describe(in_period))
        if config.TERROR_MEASURE not in terror.TERROR_MEASURES:
            data.problems.append(
                f"TERROR_MEASURE='{config.TERROR_MEASURE}' is not one of "
                f"{terror.TERROR_MEASURES}")
        needs_population = "per_100k" in config.TERROR_MEASURE
        if needs_population and data.population_raw is None:
            data.problems.append(
                f"TERROR_MEASURE='{config.TERROR_MEASURE}' needs a control "
                f"named 'population' in config.py, pointing at a file in "
                f"data/controls/.")

    print("\nCoverage inside the study period")
    header = (f"  {'variable':<22}{'rows':>8}{'countries':>11}"
              f"{'years':>7}{'filled':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for kind, tables in (("metric", data.metrics), ("control", data.controls)):
        for name, table in tables.items():
            inside = table[table["year"].between(first_year, last_year)]
            n_countries = inside["iso3"].nunique()
            n_years = inside["year"].nunique()
            possible = n_countries * (last_year - first_year + 1)
            filled = len(inside) / possible * 100 if possible else 0
            print(f"  {name:<22}{len(inside):>8,}{n_countries:>11}"
                  f"{n_years:>7}{filled:>8.0f}%")
            if inside.empty:
                data.problems.append(
                    f"{kind} '{name}': no rows inside {first_year}-{last_year}")

    print("\nWriting coverage heatmaps")
    for name, table in {**data.metrics, **data.controls}.items():
        matrix = panel.coverage_matrix(table, name, config.PERIOD)
        figure = viz.coverage_heatmap(
            matrix, name, paths.FIGURES_DIR / f"coverage_{name}.png",
            max_countries=config.MAX_COUNTRIES_IN_HEATMAP,
            dpi=config.FIGURE_DPI)
        if figure:
            print(f"  {figure.name}")

    if not data.classification.empty and data.metrics:
        print("\nHow many countries survive the merge, per metric")
        classified = set(data.classification["iso3"])
        for name, table in data.metrics.items():
            spec = config.METRICS[name]
            inside = table[table["year"].between(first_year, last_year)]
            per_country = inside.groupby("iso3")[name].count()
            enough = set(per_country[per_country >= spec.min_years].index)
            usable = enough & classified
            for control_name, control in data.controls.items():
                usable &= set(control["iso3"])
            verdict = "ok" if len(usable) >= spec.min_countries else "TOO FEW"
            print(f"  {name:<22}{len(usable):>4} countries  "
                  f"(min_countries={spec.min_countries})  {verdict}")
            if len(usable) < spec.min_countries:
                data.problems.append(
                    f"metric '{name}': only {len(usable)} countries have the "
                    f"metric, the controls and a classification at once "
                    f"(min_countries={spec.min_countries}). Loosen min_years / "
                    f"min_countries, or add data.")

    unmatched_report = countries.write_unmatched_report()
    if unmatched_report:
        print(f"\nSome country names could not be recognised. Full list in "
              f"{unmatched_report}")
        for source, names in sorted(countries.unmatched().items()):
            shown = sorted(names)[:6]
            print(f"  {source}: {len(names)} names, e.g. {shown}")
        print("  Add the real countries to data/reference/country_aliases.csv "
              "(columns: name,iso3)")

    pipeline.report_problems(data.problems)
    if data.problems:
        print("\nFix the problems above, then run this again.")
        return 1

    print("\nEverything checks out. Next: run 3_run_analysis.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
