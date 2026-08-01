"""Synthetic data with a KNOWN terrorism effect, for testing the pipeline.

The metric data for the real study is not in yet. This module invents a dataset
where the true answer is known in advance, so you can check that the machinery
recovers it before trusting it on real numbers.

Two metrics are planted:

    sample_metric   really is affected by terrorism (TRUE_TERROR_COEF), and
                    less so in developed countries (TRUE_INTERACTION)
    placebo_metric  is not affected at all - the estimate should come out
                    near zero and insignificant

Run 3_run_analysis.py --sample to analyse them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import paths, terror
from .specs import ControlSpec, FilterSpec, MetricSpec

# The truth the pipeline is supposed to recover.
TRUE_TERROR_COEF = -1.50      # effect on the developing countries
TRUE_INTERACTION = +0.90      # developed countries are hurt less
TRUE_GDP_COEF = 2.00
NOISE_SD = 1.2                # small enough that one draw recovers the truth
GDP_SHOCK_SD = 0.15           # year-to-year GDP wobble on top of the trend;
                              # without it log(GDP) is a pure country trend,
                              # which the country fixed effects nearly absorb

N_COUNTRIES = 60
N_DEVELOPED = 22
RANDOM_SEED = 20260731

SAMPLE_METRICS_DIR = paths.SAMPLE_DIR / "metrics"
SAMPLE_CONTROLS_DIR = paths.SAMPLE_DIR / "controls"
SAMPLE_FILTERS_DIR = paths.SAMPLE_DIR / "filters"
SAMPLE_TERROR_PARQUET = paths.SAMPLE_DIR / "terror_country_year.parquet"

SAMPLE_METRICS = {
    "sample_metric": MetricSpec(
        file="sample_metric.csv", column="value", min_years=8,
        min_countries=10, label="Sample metric (true effect planted)",
        unit="points"),
    "placebo_metric": MetricSpec(
        file="placebo_metric.csv", column="value", min_years=8,
        min_countries=10, label="Placebo metric (no effect planted)",
        unit="points"),
}

SAMPLE_CONTROLS = {
    "gdp_per_capita": ControlSpec(file="gdp_per_capita.csv", column="value",
                                  transform="log",
                                  label="GDP per capita (log)"),
    "population": ControlSpec(file="population.csv",
                              label="Population (log)", transform="log"),
}

SAMPLE_FILTERS = {
    "gni_per_capita": FilterSpec(
        file="gni_per_capita.csv", column="value", rule="threshold",
        threshold=13935, high_label="developed", low_label="developing",
        mode="static", label="GNI per capita", unit="current US$"),
}


def _real_iso3_codes(how_many: int) -> list:
    """Genuine ISO3 codes, so the sample goes through country resolution too."""
    import pycountry
    codes = sorted(country.alpha_3 for country in pycountry.countries)
    step = max(1, len(codes) // how_many)
    return codes[::step][:how_many]


def _fake_terror_country_year(iso3_codes, years, rng) -> pd.DataFrame:
    """A GTD-shaped country-year table, for when the real cache is not built."""
    rows = []
    for index, code in enumerate(iso3_codes):
        # Some countries are chronically hit, most are quiet: same skew as the
        # real GTD, where a handful of countries carry most of the attacks.
        base = rng.gamma(shape=0.6, scale=6.0) * (1 + 3 * (index % 7 == 0))
        for year in years:
            wave = 1.0 + 0.5 * np.sin((year - 1990) / 3.0 + index)
            attacks = rng.poisson(max(base * wave, 0.02))
            killed = rng.poisson(1.7 * attacks) if attacks else 0
            wounded = rng.poisson(2.1 * attacks) if attacks else 0
            rows.append({"iso3": code, "year": year, "n_attacks": attacks,
                         "n_attacks_success": int(attacks * 0.85),
                         "n_attacks_lethal": int(attacks * 0.4),
                         "n_attacks_suicide": int(attacks * 0.05),
                         "n_killed": killed, "n_wounded": wounded,
                         "n_casualties": killed + wounded})
    table = pd.DataFrame(rows)
    table.loc[table["year"] == terror.GTD_MISSING_YEAR,
              ["n_attacks", "n_attacks_success", "n_attacks_lethal",
               "n_attacks_suicide", "n_killed", "n_wounded",
               "n_casualties"]] = np.nan
    return terror.add_derived_measures(table)


def generate(period: Tuple[int, int] = (1990, 2021),
             use_real_terror: bool = True,
             seed: int = RANDOM_SEED) -> Dict[str, Path]:
    """Write the synthetic files into data/sample/. Returns what was written."""
    rng = np.random.default_rng(seed)
    first_year, last_year = period
    years = list(range(first_year, last_year + 1))

    for directory in (SAMPLE_METRICS_DIR, SAMPLE_CONTROLS_DIR,
                      SAMPLE_FILTERS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    terror_table = None
    if use_real_terror and paths.GTD_COUNTRY_YEAR_PARQUET.exists():
        real = pd.read_parquet(paths.GTD_COUNTRY_YEAR_PARQUET)
        busiest = (real[real["year"].between(first_year, last_year)]
                   .groupby("iso3")["n_attacks"].sum()
                   .sort_values(ascending=False).head(N_COUNTRIES).index)
        iso3_codes = sorted(busiest)
        terror_table = terror.expand_to_grid(real, iso3_codes, years)
        source = "real GTD country-year data"
    else:
        # Real ISO3 codes, so the sample exercises the same country resolution
        # as a real run does.
        iso3_codes = _real_iso3_codes(N_COUNTRIES)
        terror_table = _fake_terror_country_year(iso3_codes, years, rng)
        terror_table.to_parquet(SAMPLE_TERROR_PARQUET, index=False)
        source = "synthetic terrorism data"

    developed = set(rng.choice(iso3_codes, size=N_DEVELOPED, replace=False))

    country_effect = {code: rng.normal(50, 12) for code in iso3_codes}
    year_effect = {year: rng.normal(0, 1.5) for year in years}
    gni = {code: (rng.uniform(20_000, 70_000) if code in developed
                  else rng.uniform(600, 11_000)) for code in iso3_codes}
    gdp_base = {code: gni[code] * rng.uniform(0.85, 1.15) for code in iso3_codes}
    population_base = {code: rng.uniform(1e6, 3e8) for code in iso3_codes}

    terror_lookup = terror_table.set_index(["iso3", "year"])["log_killed"]

    metric_rows, placebo_rows, gdp_rows, gni_rows = [], [], [], []
    truth_rows = []                       # exactly what was planted, row by row
    population_wide = {}

    for code in iso3_codes:
        is_developed = code in developed
        growth = rng.uniform(0.005, 0.045)
        population_row = {}
        for offset, year in enumerate(years):
            gdp = (gdp_base[code] * (1 + growth) ** offset
                   * np.exp(rng.normal(0, GDP_SHOCK_SD)))
            people = population_base[code] * (1 + rng.uniform(0.0, 0.03)) ** offset
            population_row[year] = round(people)

            shock = terror_lookup.get((code, year), np.nan)
            if pd.isna(shock):
                metric_value = np.nan            # 1993 has no terrorism data
            else:
                metric_value = (
                    country_effect[code]
                    + year_effect[year]
                    + TRUE_GDP_COEF * np.log(gdp)
                    + TRUE_TERROR_COEF * shock
                    + (TRUE_INTERACTION * shock if is_developed else 0.0)
                    + rng.normal(0, NOISE_SD)
                )
            metric_rows.append({"country": code, "year": year,
                                "value": metric_value})
            truth_rows.append({"iso3": code, "year": year,
                               "developed": int(is_developed),
                               "shock_used": shock,
                               "planted_coef": TRUE_TERROR_COEF
                               + (TRUE_INTERACTION if is_developed else 0.0)})
            placebo_rows.append({
                "country": code, "year": year,
                "value": (country_effect[code] + year_effect[year]
                          + TRUE_GDP_COEF * np.log(gdp)
                          + rng.normal(0, NOISE_SD))})
            gdp_rows.append({"country": code, "year": year,
                             "value": round(gdp, 2)})
            # GNI stays around its own level, so the classification recovers
            # exactly the developed/developing split that was planted.
            gni_rows.append({"country": code, "year": year,
                             "value": round(gni[code] * rng.uniform(0.97, 1.03),
                                            2)})
        population_wide[code] = population_row

    written = {}

    def write_long(rows, path):
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    written["sample_metric"] = write_long(
        metric_rows, SAMPLE_METRICS_DIR / "sample_metric.csv")
    written["placebo_metric"] = write_long(
        placebo_rows, SAMPLE_METRICS_DIR / "placebo_metric.csv")
    written["gdp_per_capita"] = write_long(
        gdp_rows, SAMPLE_CONTROLS_DIR / "gdp_per_capita.csv")
    written["gni_per_capita"] = write_long(
        gni_rows, SAMPLE_FILTERS_DIR / "gni_per_capita.csv")

    # Deliberately wide, to exercise the other file layout as well.
    wide = pd.DataFrame(population_wide).T
    wide.index.name = "country"
    wide = wide.reset_index()
    path = SAMPLE_CONTROLS_DIR / "population.csv"
    wide.to_csv(path, index=False, encoding="utf-8-sig")
    written["population"] = path

    # Exactly what was planted in every row, so the recovery check is verifiable
    # rather than taken on trust.
    truth_path = paths.SAMPLE_DIR / "truth.csv"
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False,
                                    encoding="utf-8-sig")
    written["truth"] = truth_path

    print(f"Sample data written to {paths.SAMPLE_DIR} using {source}")
    print(f"  {len(iso3_codes)} countries, {len(years)} years, "
          f"{N_DEVELOPED} tagged developed")
    print(f"  planted effect: {TRUE_TERROR_COEF:+.2f} in developing countries, "
          f"{TRUE_TERROR_COEF + TRUE_INTERACTION:+.2f} in developed ones")
    return written


def load_sample_terror() -> pd.DataFrame:
    """The terrorism table the sample run should use."""
    if paths.GTD_COUNTRY_YEAR_PARQUET.exists():
        return pd.read_parquet(paths.GTD_COUNTRY_YEAR_PARQUET)
    if SAMPLE_TERROR_PARQUET.exists():
        return pd.read_parquet(SAMPLE_TERROR_PARQUET)
    raise FileNotFoundError(
        "no terrorism data available. Run 1_build_terror_cache.py, or "
        "regenerate the sample with seminar.sample_data.generate()."
    )


# A single estimate is one random draw, so it never lands exactly on the truth.
# What matters is whether it lands within a sensible number of its own standard
# errors: a strict "is the truth inside the 95% CI" test would fail one run in
# twenty even when everything is perfect.
RECOVERY_TOLERANCE_SE = 2.5


def _verdict(estimate: float, standard_error: float, truth: float) -> str:
    if not standard_error or standard_error != standard_error:
        return "no standard error"
    distance = abs(estimate - truth) / standard_error
    mark = "RECOVERED" if distance <= RECOVERY_TOLERANCE_SE else "OFF"
    return f"{distance:.1f} SE away -> {mark}"


def check_recovery(result) -> str:
    """Did the regression recover the planted effect? Used by the sample run."""
    comparison = result.comparison
    if not comparison:
        return "no estimate produced"

    estimate = comparison.get("terror_coef", float("nan"))
    standard_error = comparison.get("terror_se", float("nan"))

    if result.metric == "placebo_metric":
        low = comparison.get("terror_ci_low", float("nan"))
        high = comparison.get("terror_ci_high", float("nan"))
        inside = low <= 0 <= high
        return (f"placebo: nothing was planted. Estimate {estimate:+.3f}, "
                f"95% CI [{low:+.3f}, {high:+.3f}] "
                f"{'contains 0 as it should' if inside else 'EXCLUDES 0'}")

    lines = []
    # The pooled coefficient mixes both groups, so it is compared with the
    # planted effects averaged over how many countries are in each group.
    expected_pooled = TRUE_TERROR_COEF + TRUE_INTERACTION * (
        N_DEVELOPED / N_COUNTRIES)
    lines.append(f"pooled      planted {expected_pooled:+.3f}   estimate "
                 f"{estimate:+.3f} (SE {standard_error:.3f})   "
                 f"{_verdict(estimate, standard_error, expected_pooled)}")

    if not result.group_effects.empty:
        table = result.group_effects.set_index("group")
        for group, truth in (("developing", TRUE_TERROR_COEF),
                             ("developed", TRUE_TERROR_COEF + TRUE_INTERACTION),
                             ("difference (developed - developing)",
                              TRUE_INTERACTION)):
            if group in table.index:
                row = table.loc[group]
                label = group.split(" ")[0]
                lines.append(
                    f"{label:<11} planted {truth:+.3f}   estimate "
                    f"{row['coef']:+.3f} (SE {row['se']:.3f})   "
                    f"{_verdict(row['coef'], row['se'], truth)}")
    return "\n".join(lines)
