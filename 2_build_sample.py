# -*- coding: utf-8 -*-
"""STEP 2 - Stage 1 of the plan's data pipeline (Part 5): build the sample.

In the order File 1 §4.1 prescribes - window first, then the data rules, then
the GTD filters:

    1  File 3 Part A: every GTD event through K1-K6 and the magnitude floor
    2  File 1: every candidate country through Q1-Q6  -> the admitted sample
    3  File 2: every admitted market through C1-C3    -> developed/developing
    4  File 3 Step 1: qualifying events -> NA, NK, NI, PD per market-year

and exports what Part 5 says to carry into Stage 2:

    outputs/stage1/gtd_qualifying_events_1998_2020.csv   event level, audit
    outputs/stage1/gtd_country_year_1998_2020.csv        one row per market-year
    outputs/stage1/sample_markets.csv                    the classified sample

plus the tables and figures of §3-§5 (Figures 1, 5, 6, 8-12). The row counts
at every filter stage are printed - screenshot them (Part 5, provenance).
"""

import sys

import pandas as pd

import config
from seminar import (classification, countries, io_tabular, paths,
                     qualification, report, sources, terror, viz)
from seminar.panel import load_series
from seminar.specs import SeriesSpec


def need(path, problems):
    if not path.exists():
        problems.append(f"missing {path.relative_to(paths.PROJECT_ROOT)} - "
                        f"run 0_fetch_data.py")
        return False
    return True


def wdi(name: str) -> pd.DataFrame:
    return load_series(SeriesSpec(file=f"{name}.csv", folder="worldbank"), name)


def undp(name: str) -> pd.DataFrame:
    return load_series(SeriesSpec(file=f"{name}.csv", folder="undp"), name)


def severity_summary(q: pd.DataFrame, country_year: pd.DataFrame,
                     sample: pd.DataFrame) -> pd.DataFrame:
    """§5.3-§5.4: what the qualifying events are made of, by group."""
    groups = dict(zip(sample["iso3"], sample["group"]))
    q = q.assign(group=q["iso3"].map(groups))
    cy = country_year.assign(group=country_year["iso3"].map(groups))
    rows = []
    for group in ("developed", "developing", "all"):
        e = q if group == "all" else q[q["group"] == group]
        c = cy if group == "all" else cy[cy["group"] == group]
        casualty = (e["nkill"] >= 1) | (e["nwound"] >= 1)
        damaged = e["property"] == 1
        rows.append({
            "group": group,
            "qualifying events": len(e),
            "property only": int((damaged & ~casualty).sum()),
            "casualties only": int((casualty & ~damaged).sum()),
            "both": int((casualty & damaged).sum()),
            "catastrophic (3)": int((damaged & (e["propextent"] == 1)).sum()),
            "major (2)": int((damaged & (e["propextent"] == 2)).sum()),
            "minor (1)": int((damaged & (e["propextent"] == 3)).sum()),
            "unknown band recoded to minor": int(e["pd_unknown_recoded"].sum()),
            "property unknown (-9), scored 0": int(e["property_unknown"].sum()),
            "nkill blank": int(e["nkill_blank"].sum()),
            "nwound blank": int(e["nwound_blank"].sum()),
            "unsuccessful attacks kept": int((e["success"] == 0).sum()),
            "extended incidents": int((e["extended"] == 1).sum()),
            "market-years with NK missing": int(c["NK"].isna().sum()),
            "market-years with NI missing": int(c["NI"].isna().sum()),
            "market-years with no qualifying event": int((c["NA"] == 0).sum()),
        })
    return pd.DataFrame(rows).set_index("group").T.reset_index().rename(
        columns={"index": "measure"})


def main() -> int:
    paths.ensure_dirs()
    first, last = config.WINDOW
    problems = []
    print("=" * 72)
    print(f"STEP 2  Stage 1 - sample, classification and event aggregation   "
          f"{first}-{last}")
    print("=" * 72)

    # ---- inputs -----------------------------------------------------------
    required = [paths.GTD_EVENTS_PARQUET] + [
        paths.WORLDBANK_DIR / f"{n}.csv" for n in
        ("gdp_growth", "fdi_inflows", "fdi_outflows", "trade_openness",
         "gni_per_capita_atlas")] + [
        paths.UNDP_DIR / "hdi.csv",
        paths.CLASSIFICATION_DIR / "imf_advanced_economies.csv"]
    for path in required:
        need(path, problems)
    if problems:
        for problem in problems:
            print(f"  ! {problem}")
        if not paths.GTD_EVENTS_PARQUET.exists():
            print("  ! and run 1_build_terror_cache.py for the GTD cache")
        return 1
    try:
        events = terror.load_events()
    except terror.GTDError as error:
        print(f"\nERROR: {error}")
        return 1

    gdp_growth, fdi_in, fdi_out = (wdi("gdp_growth"), wdi("fdi_inflows"),
                                   wdi("fdi_outflows"))
    trade, gni = wdi("trade_openness"), wdi("gni_per_capita_atlas")
    gni = gni.rename(columns={"gni_per_capita_atlas": "gni_atlas"})
    hdi = undp("hdi")
    descriptors = {}
    for name in ("life_expectancy", "expected_schooling", "mean_schooling",
                 "gni_per_capita_ppp"):
        if (paths.UNDP_DIR / f"{name}.csv").exists():
            descriptors[name] = undp(name)
    history_path = paths.CLASSIFICATION_DIR / "wb_income_group_history.csv"
    income_history = None
    if history_path.exists():
        income_history = io_tabular.load_table(history_path, "income_group",
                                               numeric=False)
        income_history["year"] = income_history["year"].astype(int)
    elif config.C2_SOURCE == "history":
        print("  note: no World Bank income-group history; C2 falls back to "
              "GNI Atlas vs the configured line. Run 0_fetch_data.py.")
    imf = sources.load_imf_advanced(
        paths.CLASSIFICATION_DIR / "imf_advanced_economies.csv")
    threshold = config.C2_HIGH_INCOME_THRESHOLD
    lines = sources.load_income_thresholds(
        paths.CLASSIFICATION_DIR / "wb_income_thresholds.csv")
    if lines is not None:
        match = lines[lines["year"] == config.BASE_YEAR]
        if not match.empty:
            threshold = float(match["high_income_above"].iloc[0])

    # ---- 1. File 3 Part A, all countries -----------------------------------
    print(f"\n1. Event qualification (File 3), all GTD countries")
    world_funnel = terror.event_funnel(events, config.WINDOW,
                                       "all GTD countries")
    for _, row in world_funnel.iterrows():
        print(f"   {row['label']:<48}{int(row['events']):>9,}  "
              f"({row['pct_of_window']:5.1f}% of window)")
    qualifying_all = terror.qualifying_events(events, config.WINDOW)

    # ---- 2. File 1, Q1-Q6 ----------------------------------------------------
    print(f"\n2. Market qualification (File 1)")
    rules = qualification.QualificationRules(
        window=config.WINDOW, non_sovereign=config.NON_SOVEREIGN,
        territorial_changes=config.TERRITORIAL_CHANGES,
        min_trade_ratio=config.Q3_MIN_TRADE_RATIO,
        min_fdi_ratio=config.Q3_MIN_FDI_RATIO,
        exclusions=config.Q5_EXCLUSIONS,
        min_active_years=config.Q6_MIN_ACTIVE_YEARS,
        candidates=config.CANDIDATE_COUNTRIES)
    admitted = qualification.qualify_markets(
        events, qualifying_all, gdp_growth, fdi_in, fdi_out, trade, rules)
    for _, row in admitted.funnel.iterrows():
        drop = f"  (-{int(row['dropped'])})" if row["dropped"] else ""
        print(f"   {row['label']:<48}{int(row['remaining']):>5}{drop}")

    # ---- 3. File 2, C1-C3 ----------------------------------------------------
    print(f"\n3. Classification at the base year {config.BASE_YEAR} (File 2)")
    crules = classification.ClassificationRules(
        base_year=config.BASE_YEAR, window=config.WINDOW,
        c2_source=config.C2_SOURCE, high_income_threshold=threshold,
        index_provider_disagreements=config.INDEX_PROVIDER_DISAGREEMENTS)
    classified = classification.classify_markets(
        admitted.admitted, hdi, gni, income_history, imf, crules, descriptors)
    print(classification.summarise(classified, config.BASE_YEAR))
    sample = classified[classified["group"].notna()].reset_index(drop=True)
    if sample.empty or sample["group"].nunique() < 2:
        print("\nERROR: the sample needs both developed and developing "
              "markets. Check the qualification settings in config.py.")
        return 1

    # ---- 4. traits, flags and the market-year table ------------------------
    q = qualifying_all[qualifying_all["iso3"].isin(sample["iso3"])]
    q = terror.add_traits(q)
    q, city_notes = terror.add_city_flags(q, terror.load_cities(),
                                          config.CAPITAL_OVERRIDES)
    for note in city_notes:
        print(f"   note: {note}")
    country_year = terror.aggregate_market_years(q, sample["iso3"],
                                                 config.WINDOW)
    names = dict(zip(sample["iso3"], sample["country"]))
    groups = dict(zip(sample["iso3"], sample["group"]))
    country_year.insert(1, "country", country_year["iso3"].map(names))
    country_year.insert(2, "group", country_year["iso3"].map(groups))

    funnels = [world_funnel]
    for label, members in (("sample markets", sample["iso3"]),
                           ("developed markets",
                            sample.loc[sample["group"] == "developed",
                                       "iso3"]),
                           ("developing markets",
                            sample.loc[sample["group"] == "developing",
                                       "iso3"])):
        funnels.append(terror.event_funnel(
            events[events["iso3"].isin(members)], config.WINDOW, label))
    funnel = pd.concat(funnels, ignore_index=True)

    print("\n   What K2 and K3 remove, by group (§5.1):")
    for population in ("developed markets", "developing markets"):
        chunk = funnel[funnel["population"] == population].set_index("stage")
        print(f"   {population:<20} K2 removes "
              f"{chunk.loc['K2', 'pct_removed_at_stage']:5.1f}%, K3 removes "
              f"{chunk.loc['K3', 'pct_removed_at_stage']:5.1f}% of what "
              f"reaches it")

    # ---- exports ------------------------------------------------------------
    print("\nWriting Stage 1 exports")
    events_out = terror.export_event_table(q)
    events_out.insert(1, "group", events_out["iso3"].map(groups))
    written = [
        report.write_csv(events_out, paths.qualifying_events_csv(
            config.WINDOW).name, paths.STAGE1_DIR),
        report.write_csv(country_year, paths.country_year_csv(
            config.WINDOW).name, paths.STAGE1_DIR),
        report.write_csv(sample, paths.SAMPLE_MARKETS_CSV.name,
                         paths.STAGE1_DIR),
        report.write_csv(admitted.audit, "t3_4_market_audit.csv"),
        report.write_csv(admitted.exclusion_log, "t3_5_exclusion_log.csv"),
        report.write_csv(admitted.funnel, "t3_5_sample_funnel.csv"),
        report.write_csv(classified, "t4_7_classified_sample.csv"),
        report.write_csv(funnel, "t5_2_event_funnel.csv"),
        report.write_csv(severity_summary(q, country_year, sample),
                         "t5_4_severity_traits.csv"),
    ]
    for path in written:
        print(f"   {path.relative_to(paths.PROJECT_ROOT)}")

    # ---- figures ------------------------------------------------------------
    print("\nFigures")
    dpi = config.FIGURE_DPI
    figures = paths.FIGURES_DIR
    shares = None
    sector_files = {"agriculture": "agriculture_share",
                    "industry": "industry_share", "services": "services_share"}
    if all((paths.WORLDBANK_DIR / f"{f}.csv").exists()
           for f in sector_files.values()):
        tables = {sector: wdi(file) for sector, file in sector_files.items()}
        shares = pd.DataFrame([{
            "iso3": iso3, **{sector: classification.value_at(
                table, sector_files[sector], iso3, config.BASE_YEAR,
                config.WINDOW)[0] for sector, table in tables.items()}}
            for iso3 in sample["iso3"]])
    made = [
        viz.fig01_qualifying_attacks(country_year, sample,
                                     figures / "fig01_qualifying_attacks.png",
                                     dpi),
        viz.fig05_sectoral_composition(shares, sample, config.BASE_YEAR,
                                       figures / "fig05_sectoral_composition"
                                                 ".png", dpi)
        if shares is not None else None,
        viz.fig06_sample_funnel(admitted.funnel,
                                figures / "fig06_sample_funnel.png", dpi),
        viz.fig08_hdi_distribution(sample, config.BASE_YEAR,
                                   figures / "fig08_hdi_distribution.png", dpi),
        viz.fig09_hdi_vs_gni(sample, threshold, config.BASE_YEAR,
                             figures / "fig09_hdi_vs_gni.png", dpi),
        viz.fig10_agreement_matrix(sample, config.BASE_YEAR,
                                   figures / "fig10_agreement_matrix.png",
                                   dpi),
        viz.fig11_event_funnel(funnel, figures / "fig11_event_funnel.png",
                               dpi),
        viz.fig12_event_composition(q, sample,
                                    figures / "fig12_event_composition.png",
                                    dpi),
    ]
    for path in made:
        if path:
            print(f"   {path.relative_to(paths.PROJECT_ROOT)}")

    countries.write_unmatched_report()
    n_dev = int((sample["group"] == "developed").sum())
    print("\n" + "=" * 72)
    print(f"SAMPLE FROZEN: {len(sample)} markets ({n_dev} developed, "
          f"{len(sample) - n_dev} developing) x {last - first + 1} years = "
          f"{len(country_year):,} market-years")
    print(f"qualifying events in the sample: {len(q):,}")
    print("=" * 72)
    print("Next: 3_run_analysis.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
