# -*- coding: utf-8 -*-
"""STEP 0 - download every external series the plan needs (Part 5).

    data/worldbank/       WDI: GDP growth, FDI in/out, trade, investment,
                          inflation, GDP, GDP per capita (+ growth),
                          population, GNI per capita (Atlas), sector shares
    data/undp/            HDI and its three dimension indicators
    data/classification/  World Bank income groups by year (OGHIST) and the
                          high-income line of every year
    data/conflict/        UCDP/PRIO internal and external armed conflict
    data/reference/       capitals and largest cities (GeoNames)

The IMF advanced-economy list (data/classification/imf_advanced_economies.csv)
ships with the project: there is no API for historical WEO groupings.

Every file is an ordinary CSV you can open, check and cite. Series are
downloaded from the year before the window, because lagged controls need it.
"""

import sys

import pandas as pd

import config
from seminar import paths, sources

# Some networks (corporate proxies, antivirus with HTTPS scanning) present their
# own certificate; this makes Python use the Windows certificate store. It does
# NOT disable any security check.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


def save(table: pd.DataFrame, path) -> str:
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.relative_to(paths.PROJECT_ROOT))


def main() -> int:
    paths.ensure_dirs()
    first_year, last_year = config.WINDOW[0] - 1, config.WINDOW[1]
    failures = []

    print("=" * 72)
    print(f"STEP 0  downloading source data for {first_year}-{last_year}")
    print("=" * 72)

    print("\nWorld Bank, World Development Indicators")
    for name, (code, description) in sources.WDI_INDICATORS.items():
        print(f"  {name:<24}{code:<22}", end="", flush=True)
        try:
            table = sources.fetch_wdi(code, first_year, last_year)
        except RuntimeError as error:
            print(f"FAILED - {error}")
            failures.append(name)
            continue
        where = save(table, paths.WORLDBANK_DIR / f"{name}.csv")
        print(f"{len(table):>6,} rows {table['country'].nunique():>4} "
              f"countries -> {where}")

    print("\nUNDP, Human Development Report composite indices")
    try:
        series, label = sources.fetch_hdr(first_year, last_year)
        for name, table in series.items():
            where = save(table, paths.UNDP_DIR / f"{name}.csv")
            print(f"  {name:<24}{len(table):>6,} rows -> {where}  [{label}]")
    except RuntimeError as error:
        print(f"  FAILED - {error}")
        print("  Download it by hand from https://hdr.undp.org/data-center/"
              "documentation-and-downloads")
        failures.append("hdr")

    print("\nWorld Bank, historical income classification (OGHIST)")
    try:
        groups, thresholds = sources.fetch_oghist()
        print(f"  income groups           {len(groups):>6,} rows -> "
              f"{save(groups, paths.CLASSIFICATION_DIR / 'wb_income_group_history.csv')}")
        print(f"  high-income lines       {len(thresholds):>6,} rows -> "
              f"{save(thresholds, paths.CLASSIFICATION_DIR / 'wb_income_thresholds.csv')}")
        base = thresholds[thresholds["year"] == config.BASE_YEAR]
        if not base.empty:
            print(f"  base year {config.BASE_YEAR}: high income above US$ "
                  f"{base['high_income_above'].iloc[0]:,.0f}")
    except RuntimeError as error:
        print(f"  FAILED - {error}")
        failures.append("oghist")

    print("\nUCDP/PRIO Armed Conflict Dataset")
    try:
        conflict, label = sources.fetch_ucdp(first_year, last_year)
        where = save(conflict, paths.CONFLICT_DIR / "ucdp_conflict.csv")
        print(f"  conflict country-years  {len(conflict):>6,} rows -> {where}"
              f"  [{label}]")
    except RuntimeError as error:
        print(f"  FAILED - {error}")
        failures.append("ucdp")

    print("\nGeoNames, capitals and largest cities (for the §5.8 flags)")
    try:
        cities = sources.fetch_geonames()
        where = save(cities, paths.CITIES_CSV)
        print(f"  cities                  {len(cities):>6,} rows -> {where}")
    except RuntimeError as error:
        print(f"  FAILED - {error}")
        failures.append("geonames")

    print("\nFor the paper: cite the WDI indicator codes in "
          "seminar/sources.py, the HDR\nedition printed above, the World "
          "Bank OGHIST file, the UCDP/PRIO ACD version\nprinted above, and "
          "GeoNames (CC BY 4.0).")
    if failures:
        print(f"\n{len(failures)} download(s) failed: {failures}. Re-run; the "
              f"servers are occasionally busy.")
        return 1
    print("\nDone. Next: 1_build_terror_cache.py (once), then "
          "2_build_sample.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
