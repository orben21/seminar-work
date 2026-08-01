# -*- coding: utf-8 -*-
"""STEP 1 - convert the Global Terrorism Database into a fast cache.

Run this once (press the green arrow in PyCharm). It reads the 104 MB GTD xlsx,
keeps the twenty columns this project needs, and writes two parquet files into
data/cache/. Every later run loads those in under a second.

Takes a few minutes. You only have to do it again if you get a newer GTD file.
"""

import sys

import config
from seminar import paths, terror, viz


def main() -> int:
    paths.ensure_dirs()

    print("=" * 70)
    print("STEP 1  building the terrorism cache")
    print("=" * 70)

    try:
        events = terror.convert_to_parquet(config.GTD_XLSX,
                                           sheet_name=config.GTD_SHEET)
    except terror.GTDError as error:
        print(f"\nERROR: {error}")
        return 1

    print("\nAggregating to country-year measures")
    country_year = terror.build_country_year(
        events,
        require_all_criteria=config.GTD_REQUIRE_ALL_CRITERIA,
        drop_doubted=config.GTD_DROP_DOUBTED,
    )
    country_year.to_parquet(paths.GTD_COUNTRY_YEAR_PARQUET, index=False)
    print(f"  cached -> {paths.GTD_COUNTRY_YEAR_PARQUET}")

    print("\nSanity checks")
    print(terror.describe(country_year))

    first_year, last_year = config.PERIOD
    in_period = country_year[country_year["year"].between(first_year, last_year)]
    print(f"\nInside the study period {first_year}-{last_year}")
    print(terror.describe(in_period))

    worst = (in_period.groupby("iso3")["n_killed"].sum()
             .sort_values(ascending=False).head(10))
    print("\nMost affected countries in the period (people killed):")
    for iso3, killed in worst.items():
        print(f"  {iso3}  {int(killed):>8,}")

    figure = viz.terror_overview(in_period,
                                 paths.FIGURES_DIR / "terrorism_over_time.png",
                                 dpi=config.FIGURE_DPI)
    if figure:
        print(f"\nFigure -> {figure}")

    from seminar import countries
    report = countries.write_unmatched_report()
    if report:
        print(f"Unresolved country names listed in {report}")
        print("Add any real countries to data/reference/country_aliases.csv")

    print("\nDone. Next: put your metric files in data/metrics/, declare them "
          "in config.py, then run 2_check_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
