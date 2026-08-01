# -*- coding: utf-8 -*-
"""THE ONLY FILE YOU EDIT DAY TO DAY.

Adding data is always two steps:

    1. put the file in the right folder under data/
    2. declare it here, saying which column the program should take

Nothing is guessed unless you leave a field as None. After any change here, run
2_check_data.py - it validates every declaration against the actual files and
tells you exactly what is wrong before any analysis starts.

Folders
    data/metrics/    DEPENDENT variables. Each one gets its own regression.
    data/controls/   INDEPENDENT controls, held fixed in every model.
    data/filters/    classification data used to split developed vs developing.
"""

from seminar.specs import ControlSpec, FilterSpec, MetricSpec

# ==========================================================================
# STUDY PERIOD AND SOURCE DATA
# ==========================================================================

# Inclusive at both ends. The May 2022 GTD release stops at 2020, so 2021 has no
# terrorism data; asking for it would drop the year anyway.
PERIOD = (1990, 2020)

# The Global Terrorism Database file. It stays where it is - nothing is copied.
GTD_XLSX = (r"D:\תואר ראשון במימון ניהול עסקים ושוק ההון\שנה ב"
            r"\סמאסטר 2\סמינריון\data\globalterrorismdb_0522dist.xlsx")
GTD_SHEET = "Data"

# GTD quality filters, as normally applied in the literature.
GTD_REQUIRE_ALL_CRITERIA = True   # keep only crit1 & crit2 & crit3 events
GTD_DROP_DOUBTED = True           # drop doubtterr == 1 (may not be terrorism)


# ==========================================================================
# DEPENDENT VARIABLES - one regression per metric, run independently
# ==========================================================================
# file      : name of the file inside data/metrics/
# column    : which column holds the number (None = auto-detect)
# transform : None | "log" | "log1p" | "pct_change" | "diff" | "zscore"
# min_years : a country needs at least this many years to be included
# label/unit: how it is captioned in tables and figures

METRICS = {
    "trade_openness": MetricSpec(
        file="trade_openness.csv",
        column="value",
        transform=None,
        min_years=10,
        label="Trade openness",
        unit="% of GDP",
    ),
    "fdi_inflows": MetricSpec(
        file="fdi_inflows.csv",
        column="value",
        transform=None,
        min_years=10,
        label="FDI net inflows",
        unit="% of GDP",
    ),
    "fdi_outflows": MetricSpec(
        file="fdi_outflows.csv",
        column="value",
        transform=None,
        min_years=10,
        label="FDI net outflows",
        unit="% of GDP",
    ),
    "gdp_growth": MetricSpec(
        file="gdp_growth.csv",
        column="value",
        transform=None,
        min_years=10,
        label="GDP per capita growth",
        unit="% per year",
    ),
    "market_cap": MetricSpec(
        file="market_cap.csv",
        column="value",
        transform=None,
        min_years=10,
        label="Stock market capitalisation",
        unit="% of GDP",
    ),
}


# ==========================================================================
# INDEPENDENT CONTROLS - held fixed so terrorism is not credited for them
# ==========================================================================

CONTROLS = {
    "gdp_per_capita": ControlSpec(
        file="gdp_per_capita.csv",
        column="value",
        transform="log",
        label="GDP per capita (log)",
    ),
    "population": ControlSpec(
        file="population.csv",
        column="value",
        transform="log",
        label="Population (log)",
    ),
    # "inflation": ControlSpec(file="inflation.csv", column="value",
    #                          label="Inflation, CPI"),
}


# ==========================================================================
# FILTRATION - how a country is tagged developed or developing
# ==========================================================================
# rule="threshold"   : value >= threshold -> high_label, else low_label
# rule="categorical" : the text value is mapped through `categories`
# mode="static"      : one fixed tag per country (recommended - the group never
#                      changes, so the interaction coefficient is interpretable)
# mode="time_varying": the tag may change from year to year

FILTERS = {
    "gni_per_capita": FilterSpec(
        file="gni_per_capita.csv",
        column="value",
        rule="threshold",
        threshold=13935,           # World Bank high-income line, current US$
        high_label="developed",
        low_label="developing",
        mode="static",
        static_stat="median",
        label="GNI per capita",
        unit="current US$",
    ),
    "hdi": FilterSpec(
        file="hdi.csv",            # download from hdr.undp.org and drop it in
        column="value",
        rule="threshold",
        threshold=0.800,           # UNDP "very high human development" line
        high_label="developed",
        low_label="developing",
        mode="static",
        static_stat="median",
        label="Human Development Index",
    ),
    "income_group": FilterSpec(
        file="income_group.csv",
        column="value",
        rule="categorical",
        categories={
            "High income": "developed",
            "Upper middle income": "developing",
            "Lower middle income": "developing",
            "Low income": "developing",
        },
        mode="static",
        label="World Bank income group",
    ),
}

ACTIVE_FILTER = "gni_per_capita"   # which classification THIS run uses

COUNTRY_SUBSET = None              # None = every country that has data,
                                   # or a list like ["USA", "GBR", "IND"]


# ==========================================================================
# THE TERRORISM SIDE
# ==========================================================================
# Any column the country-year terrorism table produces:
#   n_attacks  n_attacks_success  n_attacks_lethal  n_attacks_suicide
#   n_killed   n_wounded          n_casualties
#   log_attacks  log_killed  log_wounded  log_casualties
#   terror_any   terror_intensity
#   attacks_per_100k  killed_per_100k        (need a "population" control)
#   log_attacks_per_100k  log_killed_per_100k
TERROR_MEASURE = "log_killed"

# Distributed lag structure. [0] alone = same-year effect only.
# [0, 1, 2, 3] also estimates the cumulative effect over four years, which is
# the "long term" claim of the paper.
LAGS = [0, 1, 2, 3]


# ==========================================================================
# OUTPUT
# ==========================================================================

FIGURE_DPI = 300                   # 300 is print quality for the paper
FIGURE_FORMAT = "png"
MAX_COUNTRIES_IN_HEATMAP = 60      # coverage heatmaps get unreadable past this
