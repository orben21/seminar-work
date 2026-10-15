# -*- coding: utf-8 -*-
"""STEP 3 - Stage 2 of the plan's pipeline (Part 5): the analysis.

Reads only what Stage 1 exported plus the World Bank / UNDP / UCDP extracts -
the same inputs Part 5 says go into the conversation - and then:

    §5.6  builds the Terrorism Severity Index on the frozen sample
    §6    assembles the panel: TSI(t), TSI(t-1), outcomes, controls, group
    §6.6  descriptive statistics and correlations
    §7.4  diagnostics, which choose the standard errors
    §7.1  the four regressions x two outcomes (H1: GDP growth, H2: FDI ratio)
    §8    build-up tables, marginal effects, component decomposition, the
          hypothesis-test summary, Figures 13A-24 and outputs/report.html

    --sample   replace the outcomes with synthetic ones carrying a KNOWN TSI
               effect, to prove the regressions recover it
    --open     open the report when finished
"""

import argparse
import copy
import sys
import webbrowser

import pandas as pd

import config
from seminar import (descriptives, diagnostics, paths, regression, report,
                     sample_data, tsi, viz)
from seminar.panel import (ALL_MARKETS_SUFFIX, INTERACTION, INTERACTION_LAG,
                           TREATMENT, TREATMENT_LAG, build_panel,
                           check_series)
from seminar.regression import SE_CLUSTER, SE_LABELS

READING = ("How to read every coefficient below (§5.6): the TSI is standardised "
           "on this sample, so a coefficient answers <b>what change in the "
           "outcome is associated with a one-standard-deviation rise in "
           "terrorism severity</b> - not one more attack, not one more death, "
           "not terrorism versus none. Magnitudes are therefore not comparable "
           "with studies that use raw counts or dummies, and no coefficient is "
           "a per-attack effect.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true",
                        help="synthetic outcomes with a planted TSI effect")
    parser.add_argument("--open", action="store_true",
                        help="open the HTML report when finished")
    return parser.parse_args()


def read_stage1():
    files = {"country_year": paths.country_year_csv(config.WINDOW),
             "sample": paths.SAMPLE_MARKETS_CSV,
             "events": paths.qualifying_events_csv(config.WINDOW)}
    missing = [p for p in files.values() if not p.exists()]
    if missing:
        return None, missing
    loaded = {k: pd.read_csv(p, encoding="utf-8-sig") for k, p in
              files.items()}
    return loaded, []


def stage1_table(name):
    path = paths.TABLES_DIR / name
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


def spot_checks(country_year, events, n, seed) -> pd.DataFrame:
    """Appendix E: market-years small enough to verify by hand in the raw GTD."""
    candidates = country_year[country_year["NA"].between(1, 25)]
    picked = candidates.sample(n=min(n, len(candidates)), random_state=seed)
    rows = []
    for _, row in picked.sort_values(["iso3", "year"]).iterrows():
        ids = events[(events["iso3"] == row["iso3"])
                     & (events["iyear"] == row["year"])]["eventid"]
        rows.append({"iso3": row["iso3"], "country": row["country"],
                     "year": int(row["year"]), "NA": row["NA"],
                     "NK": row["NK"], "NI": row["NI"], "PD": row["PD"],
                     "eventids": " ".join(str(int(i)) for i in ids),
                     "checked_by_hand": ""})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()
    first, last = config.WINDOW
    print("=" * 72)
    print(f"STEP 3  Stage 2 - TSI and panel regressions   {first}-{last}"
          + ("   [SAMPLE MODE]" if args.sample else ""))
    print("=" * 72)

    stage1, missing = read_stage1()
    if stage1 is None:
        for path in missing:
            print(f"  ! missing {path.relative_to(paths.PROJECT_ROOT)}")
        print("Run 2_build_sample.py first.")
        return 1
    problems = check_series(config.OUTCOMES, config.CONTROLS)
    if problems:
        for problem in problems:
            print(f"  ! {problem}")
        return 1
    sample, events = stage1["sample"], stage1["events"]
    country_year = stage1["country_year"]
    dpi = config.FIGURE_DPI
    if args.sample:                  # never write over the real results
        tables_dir = paths.SAMPLE_CHECK_DIR / "tables"
        figures = paths.SAMPLE_CHECK_DIR / "figures"
        report_path = paths.SAMPLE_CHECK_DIR / "report.html"
    else:
        tables_dir, figures = paths.TABLES_DIR, paths.FIGURES_DIR
        report_path = paths.REPORT_HTML

    # ---- §5.6 the TSI -------------------------------------------------------
    print(f"\n§5.6  TSI on {sample['iso3'].nunique()} markets x "
          f"{last - first + 1} years = {len(country_year):,} market-years")
    index = tsi.build_tsi(country_year)
    raw_skew = (index.skewness[index.skewness["stage"] == "raw"]
                .set_index("component")["skewness"])
    for _, row in index.moments.iterrows():
        print(f"   {row['component']}: ln(1+x) mean {row['mean_ln1p']:.3f}, "
              f"sd {row['sd_ln1p']:.3f}   (raw skewness "
              f"{raw_skew[row['component']]:.2f})")
    partial = int((index.table["n_components"] < 4).sum())
    if partial:
        print(f"   {partial} market-years have a missing component (every "
              f"event's count blank) and average the components they have")
    worked = tsi.worked_example(index.table, index.moments, events)
    by_market = tsi.skewness_by_market(index.table, "NK")

    # ---- §6 the panel -----------------------------------------------------
    panel = build_panel(index.table, sample, config.OUTCOMES, config.CONTROLS,
                        config.WINDOW)
    if args.sample:
        panel = sample_data.plant(panel)
    for note in panel.notes:
        print(f"   note: {note}")
    outcomes = panel.outcomes
    labels = {name: spec.label for name, spec in outcomes.items()}
    labels.update({name: spec.label or name
                   for name, spec in config.CONTROLS.items()})
    labels.update({"TSI": "TSI", "NA": "NA attacks", "NK": "NK killed",
                   "NI": "NI injured", "PD": "PD property"})

    # ---- §6.6 descriptives --------------------------------------------------
    controls = panel.control_names
    variables = list(outcomes) + ["TSI", "NA", "NK", "NI", "PD"] + controls
    described = descriptives.describe_by_group(panel.data, variables, labels)
    corr_vars = list(outcomes) + ["TSI", TREATMENT_LAG] + controls
    correlations = descriptives.correlation_matrix(panel.data, corr_vars)
    extremes = {name: descriptives.extreme_values(panel.data, name)
                for name in outcomes}

    # ---- §7 per outcome -----------------------------------------------------
    core, effects, buildups, components, diag, vifs, robust = \
        {}, {}, {}, {}, [], [], {}
    se_types = {}
    borderline = sample.loc[sample["borderline"].astype(bool), "iso3"].tolist()
    for name, spec in outcomes.items():
        print(f"\n§7  {spec.hypothesis}: {spec.label}")
        base = regression.estimation_sample(panel, name)
        terms = [TREATMENT, TREATMENT_LAG] + controls
        first_pass = regression.fit_fe(base, name, terms, SE_CLUSTER)
        tests, vif, se_type = diagnostics.run_diagnostics(
            base, name, terms, first_pass, config.SE_TYPE)
        diag.append(tests)
        vifs.append(vif.assign(outcome=name))
        se_types[name] = se_type
        print(f"   standard errors: {SE_LABELS[se_type]} "
              f"({tests['note'].iloc[-1]})")

        core[name] = regression.core_regressions(panel, name, se_type)
        fit4 = core[name].get("4_interaction")
        effects[name] = (regression.marginal_effects(fit4) if fit4
                         else pd.DataFrame())
        buildups[name] = regression.buildup(panel, name, se_type)
        components[name] = regression.component_regressions(
            panel, name, se_type, config.COMPONENT_ENTRY)
        if config.ROBUSTNESS_EXCLUDE_BORDERLINE and borderline:
            robust[name] = regression.core_regressions(panel, name, se_type,
                                                       exclude=borderline)
        for key, fit in core[name].items():
            row = fit.get(TREATMENT)
            extra = ""
            if key == "4_interaction":
                b3 = fit.get(INTERACTION)
                extra = (f"   b3 {b3['coef']:+.3f}{b3['stars']} "
                         f"(p={b3['p']:.3f})")
            print(f"   {fit.title:<42} TSI(t) {row['coef']:+.3f}{row['stars']:<3}"
                  f" (p={row['p']:.3f}, N={fit.n_obs}){extra}")

    # ---- §8.5 check: an outcome's regressions with every market ---------
    # OutcomeSpec.exclude_mean_above leaves some markets out of one outcome's
    # regressions (the FDI ratio's financial centres). The same four
    # regressions are re-run with them, same standard errors, and reported.
    with_all = {}
    for name, dropped in getattr(panel, "excluded", {}).items():
        if name not in outcomes:
            continue
        alt = copy.copy(panel)
        alt.data = panel.data.assign(
            **{name: panel.data[name + ALL_MARKETS_SUFFIX]})
        fits = regression.core_regressions(alt, name, se_types[name])
        tag = ", ".join(dropped)
        for fit in fits.values():
            fit.title = f"{fit.title} [check: with {tag}]"
        with_all[name] = fits
        print(f"\n§8.5 check  {outcomes[name].label} with {tag} included")
        for key, fit in fits.items():
            row = fit.get(TREATMENT)
            extra = ""
            if key == "4_interaction":
                b3 = fit.get(INTERACTION)
                extra = (f"   b3 {b3['coef']:+.3f}{b3['stars']} "
                         f"(p={b3['p']:.3f})")
            print(f"   {fit.title[:42]:<42} TSI(t) {row['coef']:+.3f}"
                  f"{row['stars']:<3} (p={row['p']:.3f}, N={fit.n_obs}){extra}")

    hypotheses = {name: spec.hypothesis for name, spec in outcomes.items()}
    tests_table = regression.hypothesis_tests(core, hypotheses)
    all_fits = [f for fits in core.values() for f in fits.values()] \
        + [f for fits in buildups.values() for f in fits] \
        + [f for fits in components.values() for f in fits] \
        + [f for fits in robust.values() for f in fits.values()] \
        + [f for fits in with_all.values() for f in fits.values()]

    # ---- tables -------------------------------------------------------------
    print("\nWriting tables")
    core_terms = [TREATMENT, TREATMENT_LAG, INTERACTION, INTERACTION_LAG] \
        + controls
    split_rows, interaction_tables, buildup_tables = [], {}, {}
    for name in outcomes:
        fits = core[name]
        split = [fits[k] for k in ("2_developed", "3_developing") if k in fits]
        split_rows.append(regression.coefficient_columns(
            split, [TREATMENT, TREATMENT_LAG] + controls).assign(outcome=name))
        interaction_tables[name] = regression.coefficient_columns(
            [fits[k] for k in ("1_pooled", "4_interaction") if k in fits],
            core_terms)
        buildup_tables[name] = regression.coefficient_columns(
            buildups[name], [TREATMENT, TREATMENT_LAG] + controls)
    component_table = pd.concat([regression.component_table(components[n])
                                 for n in outcomes], ignore_index=True)
    marginal = pd.concat(effects.values(), ignore_index=True)
    tables = {
        "t5_6_tsi_moments.csv": index.moments,
        "t5_6_skewness.csv": index.skewness,
        "t5_6_nk_skewness_by_market.csv": by_market,
        "t5_6_worked_example.csv": worked,
        "t5_6_tsi_panel.csv": index.table,
        "t6_6_descriptives.csv": described,
        "t6_6_correlations.csv": correlations.reset_index().rename(
            columns={"index": "variable"}),
        "t6_6_extreme_values.csv": pd.concat(extremes.values(),
                                             ignore_index=True),
        "t8_3_diagnostics.csv": pd.concat(diag, ignore_index=True),
        "t8_3_vif.csv": pd.concat(vifs, ignore_index=True),
        "t8_6_split_sample.csv": pd.concat(split_rows, ignore_index=True),
        "t8_6_marginal_effects.csv": marginal,
        "t8_7_components.csv": component_table,
        "t8_8_hypothesis_tests.csv": tests_table,
        "appD_all_coefficients.csv": regression.all_coefficients(all_fits),
        "appD_models.csv": regression.model_table(all_fits),
        "appE_spot_checks.csv": spot_checks(country_year, events,
                                            config.SPOT_CHECK_MARKET_YEARS,
                                            config.RANDOM_SEED),
    }
    for name in outcomes:
        section = "8_4" if outcomes[name].hypothesis == "H1" else "8_5"
        tables[f"t{section}_{name}_buildup.csv"] = buildup_tables[name]
        tables[f"t8_6_{name}_interaction.csv"] = interaction_tables[name]
    for name, fits in with_all.items():
        tables[f"t8_5_{name}_check_all_markets.csv"] = \
            regression.coefficient_columns(list(fits.values()), core_terms)
        if "4_interaction" in fits:
            tables[f"t8_5_{name}_check_all_markets_marginal.csv"] = \
                regression.marginal_effects(fits["4_interaction"])
    if robust:
        tables["t4_5_robustness_without_borderline.csv"] = pd.concat(
            [regression.coefficient_columns(list(robust[n].values()),
                                            core_terms).assign(outcome=n)
             for n in robust], ignore_index=True)
    for file_name, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            report.write_csv(table, file_name, tables_dir)
    print(f"   {len(tables)} tables in "
          f"{tables_dir.relative_to(paths.PROJECT_ROOT)}")

    # ---- figures ------------------------------------------------------------
    print("Drawing figures")
    fig = {}
    fig["13a"] = viz.fig13a_skewness_by_market(
        by_market, sample, figures / "fig13a_nk_skewness_by_market.png", dpi)
    fig["14"] = viz.fig14_transformation(index.table,
                                         figures / "fig14_transformation.png",
                                         dpi)
    fig["15"] = viz.fig15_tsi_heatmap(index.table, sample,
                                      figures / "fig15_tsi_heatmap.png", dpi)
    fig["16"] = viz.fig16_correlation_matrix(
        correlations, labels, figures / "fig16_correlation_matrix.png", dpi)
    fig["17"] = viz.fig17_components_over_time(
        index.table, sample, figures / "fig17_components_over_time.png", dpi)
    fig["18"] = viz.fig18_tsi_distribution(
        index.table, sample, figures / "fig18_tsi_distribution.png", dpi)
    fig["19"] = viz.fig19_outcomes_over_time(
        panel.data, outcomes, figures / "fig19_outcomes_over_time.png", dpi)
    for number, name in zip((20, 21), outcomes):
        fig[str(number)] = viz.fig20_21_coefplot(
            buildups[name], outcomes[name].label, outcomes[name].unit, number,
            figures / f"fig{number}_coefplot_{name}.png", dpi)
    if not marginal.empty:
        fig["22"] = viz.fig22_marginal_effects(
            marginal, labels, figures / "fig22_marginal_effects.png", dpi)
    fig["23"] = viz.fig23_binned_scatter(
        {name: regression.estimation_sample(panel, name) for name in outcomes},
        outcomes, controls, figures / "fig23_binned_scatter.png", dpi=dpi)
    if not component_table.empty:
        fig["24"] = viz.fig24_component_coefficients(
            component_table, labels,
            figures / "fig24_component_coefficients.png", dpi)

    # ---- report -------------------------------------------------------------
    build_report(args, sample, index, worked, described, extremes, diag, vifs,
                 buildup_tables, split_rows, interaction_tables, marginal,
                 component_table, tests_table, robust, core, panel,
                 figures, tables_dir, report_path)

    # ---- console summary ----------------------------------------------------
    print("\n" + "=" * 72)
    print("HYPOTHESIS TESTS (§8.8)" + ("  [SYNTHETIC]" if args.sample else ""))
    print("=" * 72)
    for _, row in tests_table.iterrows():
        coef = "" if pd.isna(row["coef"]) else f"{row['coef']:+.3f} "
        print(f"{row['hypothesis']} {row['quantity']:<46} {coef}"
              f"p={row['p']:.3f}  {row['decision']}")
    for name, fits in with_all.items():
        fit4 = fits.get("4_interaction")
        if fit4 is not None:
            b3 = fit4.get(INTERACTION)
            print(f"{outcomes[name].hypothesis} check, all markets: "
                  f"b3 {b3['coef']:+.3f} p={b3['p']:.3f}  (N={fit4.n_obs})")
    if args.sample:
        print("\nDID THE REGRESSIONS RECOVER THE PLANTED EFFECT?")
        for line in sample_data.check_recovery(core):
            print(f"  {line}")
    print(f"\nFull report: {report_path}")
    if args.open:
        webbrowser.open(report_path.as_uri())
    return 0


def build_report(args, sample, index, worked, described, extremes, diag, vifs,
                 buildup_tables, split_rows, interaction_tables, marginal,
                 component_table, tests_table, robust, core, panel,
                 figures, tables_dir, report_path):
    stage1 = paths.FIGURES_DIR          # §3-§5 figures, drawn by Stage 1
    S = report.Section
    sections = []

    s = S("reading", "How to read the results")
    s.reading(READING)
    if args.sample:
        s.note("SAMPLE MODE: the outcomes are synthetic, with a known effect "
               "planted (see the console). Nothing here is a finding.")
    sections.append(s)

    s = S("s3", "§3.5 The admitted sample (File 1)")
    s.figure(stage1 / "fig06_sample_funnel.png", "Figure 6")
    s.table(stage1_table("t3_5_exclusion_log.csv"),
            "Appendix C - exclusion log: the first rule each candidate failed "
            "and what triggered it (t3_5_exclusion_log.csv; every rule for "
            "every candidate in t3_4_market_audit.csv)")
    sections.append(s)

    s = S("s4", "§4 Developed vs developing (File 2)")
    master = stage1_table("t4_7_classified_sample.csv")
    if master is not None:
        s.table(master[["country", "iso3", "hdi", "hdi_year", "hdi_tier",
                        "gni_atlas", "income_group", "imf_group",
                        "undp_group", "wb_group", "c3_agreements", "group",
                        "borderline", "borderline_triggers"]],
                "§4.7 master table (t4_7_classified_sample.csv)")
    for number, name in (("8", "fig08_hdi_distribution.png"),
                         ("9", "fig09_hdi_vs_gni.png"),
                         ("10", "fig10_agreement_matrix.png"),
                         ("5", "fig05_sectoral_composition.png")):
        s.figure(stage1 / name, f"Figure {number}")
    sections.append(s)

    s = S("s5", "§5.2-§5.4 Qualifying events (File 3, Part A)")
    s.table(stage1_table("t5_2_event_funnel.csv"),
            "Figure 11 data - events remaining after each stage")
    s.figure(stage1 / "fig11_event_funnel.png", "Figure 11")
    s.table(stage1_table("t5_4_severity_traits.csv"),
            "Severity traits of the qualifying events (§5.3-§5.4)")
    s.figure(stage1 / "fig12_event_composition.png", "Figure 12")
    s.figure(stage1 / "fig01_qualifying_attacks.png", "Figure 1")
    sections.append(s)

    s = S("s56", "§5.6 The Terrorism Severity Index")
    s.table(index.moments, "Appendix D - the pooled means and standard "
                           "deviations; any TSI value can be rebuilt from them")
    s.table(index.skewness, "Appendix D - shape of each component before and "
                            "after the log; the z column keeps the log's "
                            "skewness because standardising is linear")
    s.table(worked, "Figure 13 worked example - one market-year from raw "
                    "aggregates to the TSI")
    s.figure(figures / "fig13a_nk_skewness_by_market.png", "Figure 13A")
    s.figure(figures / "fig14_transformation.png", "Figure 14")
    sections.append(s)

    s = S("s66", "§6.6 Descriptive statistics")
    s.table(described, "By group, with the difference in means and a p-value "
                       "clustered by market (t6_6_descriptives.csv)")
    for name, table in extremes.items():
        s.table(table, f"Most extreme market-years of {name} - check they are "
                       f"not carrying the estimate")
    s.figure(figures / "fig15_tsi_heatmap.png", "Figure 15")
    s.figure(figures / "fig16_correlation_matrix.png", "Figure 16")
    for note in panel.notes:
        s.note(note)
    sections.append(s)

    s = S("s81", "§8.1-§8.2 Descriptive patterns")
    s.figure(figures / "fig17_components_over_time.png", "Figure 17")
    s.figure(figures / "fig18_tsi_distribution.png", "Figure 18")
    s.figure(figures / "fig19_outcomes_over_time.png", "Figure 19")
    sections.append(s)

    s = S("s83", "§8.3 Diagnostic tests (§7.4)")
    s.table(pd.concat(diag, ignore_index=True),
            "Every test on regression 1 of each outcome; the last rows give "
            "the standard errors adopted")
    s.table(pd.concat(vifs, ignore_index=True), "Variance inflation factors, "
                                                "raw and within the fixed "
                                                "effects")
    sections.append(s)

    for number, name in zip((20, 21), panel.outcomes):
        spec = panel.outcomes[name]
        section = "8.4" if spec.hypothesis == "H1" else "8.5"
        s = S(f"s{section}", f"§{section} {spec.hypothesis} - terrorism and "
                             f"{spec.label}")
        s.reading(READING)
        s.table(buildup_tables[name], "Columns: TSI only, + controls, + year "
                                      "FE, + lag - same rows; country FE "
                                      "throughout")
        s.figure(figures / f"fig{number}_coefplot_{name}.png",
                 f"Figure {number}")
        dropped = getattr(panel, "excluded", {}).get(name, {})
        check = tables_dir / f"t8_5_{name}_check_all_markets.csv"
        if dropped and check.exists():
            names = ", ".join(f"{k} (mean {v:.0f})" for k, v in
                              dropped.items())
            s.note(f"The regressions above leave out {names}: their mean "
                   f"{spec.label} over the window is above "
                   f"{spec.exclude_mean_above:g} ({spec.unit}). The check "
                   f"below re-runs regressions 1-4 with them included.")
            s.table(pd.read_csv(check, encoding="utf-8-sig"),
                    f"Check: regressions 1-4 for {spec.label} with every "
                    f"market included")
            marg = tables_dir / f"t8_5_{name}_check_all_markets_marginal.csv"
            if marg.exists():
                s.table(pd.read_csv(marg, encoding="utf-8-sig"),
                        "Check: marginal effects with every market included")
        sections.append(s)

    s = S("s86", "§8.6 The developed / developing difference")
    s.table(pd.concat(split_rows, ignore_index=True),
            "§7.3(a) split sample: regressions 2 and 3 side by side")
    for name, table in interaction_tables.items():
        s.table(table, f"§7.3(b) {name}: regression 1 and regression 4. "
                       f"b3 (TSI_x_dev) is the formal test of "
                       f"{panel.outcomes[name].hypothesis}")
    s.table(marginal, "Marginal effects from regression 4: developing = b1, "
                      "developed = b1 + b3, difference = b3")
    s.figure(figures / "fig22_marginal_effects.png",
             "Figure 22 - the headline figure")
    s.figure(figures / "fig23_binned_scatter.png", "Figure 23")
    sections.append(s)

    s = S("s87", "§8.7 Component decomposition")
    s.table(component_table, f"Each component's z(ln(1+x)) in place of the "
                             f"TSI (mode: {config.COMPONENT_ENTRY})")
    s.figure(figures / "fig24_component_coefficients.png", "Figure 24")
    sections.append(s)

    s = S("s88", "§8.8 Summary of hypothesis tests")
    s.table(tests_table, "Two-sided tests at 5% (Part 0.4)")
    robust_path = tables_dir / "t4_5_robustness_without_borderline.csv"
    if robust and robust_path.exists():
        s.table(pd.read_csv(robust_path, encoding="utf-8-sig"),
                "Robustness R5: regressions without the borderline markets")
    notes = sorted({note for fits in core.values() for fit in fits.values()
                    for note in fit.notes})
    for note in notes:
        s.note(note)
    sections.append(s)

    run_info = [
        ("window", f"{config.WINDOW[0]}-{config.WINDOW[1]}"),
        ("base year (classification)", str(config.BASE_YEAR)),
        ("markets", f"{len(sample)} ({int((sample['group'] == 'developed').sum())}"
                    f" developed, "
                    f"{int((sample['group'] == 'developing').sum())} "
                    f"developing)"),
        ("outcomes", ", ".join(f"{s.hypothesis}: {s.label}"
                               for s in panel.outcomes.values())),
        ("controls", ", ".join(panel.control_names)),
        ("standard errors", config.SE_TYPE),
        ("mode", "SAMPLE (synthetic outcomes)" if args.sample else "real data"),
    ]
    path = report.build_report(
        "Terrorism, GDP growth and FDI in developed and developing countries",
        "Stage 1 sample and Stage 2 panel regressions, following the action "
        "plan and Files 1-3", run_info, sections, path=report_path)
    print(f"   report -> {path.relative_to(paths.PROJECT_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
