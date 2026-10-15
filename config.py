# -*- coding: utf-8 -*-
"""THE ONLY FILE YOU EDIT DAY TO DAY.

Every setting the study depends on, grouped by the document that governs it:

    PART 0   observation window and base year        (action plan, Part 0)
    FILE 1   market qualification Q1-Q6              -> seminar/qualification.py
    FILE 2   developed / developing, C1-C3           -> seminar/classification.py
    FILE 3   event filters K1-K6, floor, TSI         -> seminar/terror.py, tsi.py
    §6       outcomes and controls                   -> seminar/panel.py
    §7       estimation                              -> seminar/regression.py

What Files 1-3 fix exactly - K2 = all three criteria, K3 = doubtterr 0, the
magnitude floor, the propextent weights 3/2/1/1, the TSI formula, HDI >= 0.800
and its 0.02 borderline band - is coded in those modules next to its source,
and is deliberately NOT a switch here: changing it would change the study's
definitions. What lives here are the choices the Files leave open. Each one
has to be stated in the methodology chapter.

After any change, run 2_build_sample.py: the sample and the TSI pool are rebuilt
from scratch, because the TSI depends on exactly which markets are in it
(plan §5.6).
"""

from seminar.specs import ControlSpec, OutcomeSpec

# ==========================================================================
# PART 0 - FIXED PARAMETERS
# ==========================================================================

# W1, 1998-2020 (Part 0.1). Cannot open before 1998: File 3 §3.2 - GTD coded
# doubtterr routinely only after 1997, so K3 would drop earlier events for
# missing coding rather than for failing the definition. The May 2022 GTD
# release ends in 2020; its collection breaks (2008, 2012) are absorbed by
# the year fixed effects.
WINDOW = (1998, 2020)

# Every market is classified once, at the base year, and keeps its group for
# the whole panel (File 2 §6.1).
BASE_YEAR = 1998

# The Global Terrorism Database file. It stays where it is - nothing is copied.
GTD_XLSX = (r"E:\הקריה האקדמית תואר ראשון\שנה ב\סמאסטר 2\סמינריון\DATA"
            r"\globalterrorismdb_0522dist.xlsx")
GTD_SHEET = "Data"


# ==========================================================================
# FILE 1 - WHICH MARKETS ENTER THE SAMPLE (Q1-Q6)
# ==========================================================================

# None = every country the GTD records in the window, which is where Figure 6
# starts ("all countries in GTD"). Or an explicit list of ISO3 codes - for
# example a published sample - to which Q1-Q6 are then applied.
CANDIDATE_COUNTRIES = None

# Q1 - a market is the economy of a SOVEREIGN statistical unit (File 1 §2).
# Territories the GTD or the World Bank report separately fail Q1.
NON_SOVEREIGN = {
    "HKG": "Special Administrative Region of China",
    "MAC": "Special Administrative Region of China",
    "PRI": "US territory",
    "NCL": "French overseas collectivity",
    "PYF": "French overseas collectivity",
    "GLP": "French overseas department",
    "MTQ": "French overseas department",
    "GUF": "French overseas department",
    "REU": "French overseas department",
    "WLF": "French overseas collectivity",
    "FLK": "British overseas territory",
    "ESH": "non-self-governing territory",
    "BMU": "British overseas territory",
    "CYM": "British overseas territory",
    "VGB": "British overseas territory",
    "TCA": "British overseas territory",
    "GIB": "British overseas territory",
    "IMN": "British Crown dependency",
    "ABW": "constituent country of the Netherlands",
    "CUW": "constituent country of the Netherlands",
    "SXM": "constituent country of the Netherlands",
    "MAF": "French overseas collectivity",
    "GUM": "US territory",
    "VIR": "US territory",
    "ASM": "US territory",
    "MNP": "US territory",
    "GRL": "autonomous territory of Denmark",
    "FRO": "autonomous territory of Denmark",
}

# Q1 - "a single, consistently named unit ... over the whole observation
# window". Countries whose territory changed inside 1998-2020. (Countries the
# GTD names more than one way inside the window - Yugoslavia / Serbia-
# Montenegro / Serbia - are caught automatically.)
TERRITORIAL_CHANGES = {
    "SDN": "South Sudan seceded in July 2011",
    "SSD": "independent only from July 2011",
    "SRB": "Serbia and Montenegro separated in 2006; Kosovo declared "
           "independence in 2008",
    "MNE": "independent only from June 2006",
    "XKX": "UN administration from 1999, independence declared 2008",
    "TLS": "independent only from May 2002",
}

# Q3 - "a non-trivial Trade Volume Ratio and a non-trivial FDI Ratio".
# File 1 does not quantify "non-trivial"; these are the study's choice (mean
# over the window, % of GDP). State them in §3.4.
Q3_MIN_TRADE_RATIO = 10.0
Q3_MIN_FDI_RATIO = 0.5

# Q5 - exclusion for data integrity: series unavailable, interrupted or
# unreliable because of active war or contested statehood. Seeded with the
# precedent File 1 cites: Saeed et al. (2018) excluded Afghanistan, Iraq and
# Palestine. Q5 is NOT a severity filter (File 1 §4.1) - never add a country
# here because it suffered a lot of terrorism.
Q5_EXCLUSIONS = {
    "AFG": "active war; excluded as an outlier by Saeed et al. (2018)",
    "IRQ": "active war; excluded as an outlier by Saeed et al. (2018)",
    "PSE": "contested statehood; excluded by Saeed et al. (2018)",
}

# Q6 - "so few qualifying events that no within-country variation exists".
# Operationalised as the number of window years with at least one qualifying
# event. File 1 gives no number; this is the study's choice. State it in §3.4.
Q6_MIN_ACTIVE_YEARS = 3


# ==========================================================================
# FILE 2 - DEVELOPED OR DEVELOPING (C1-C3)
# ==========================================================================

# C2 - which World Bank assignment to use at the base year:
#   "history"  the World Bank's own classification of that year, from its
#              historical table (OGHIST) - "the income-group assignment of the
#              matching vintage" (plan §4.3). Recommended.
#   "gni"      GNI per capita (Atlas) of the base year against the line below
C2_SOURCE = "history"

# High-income line for the base year's GNI, used for Figure 9 and whenever the
# historical assignment is unavailable. 9,360 US$ is the FY2000 line, set on
# 1998 GNI (OGHIST). 0_fetch_data.py downloads the whole threshold history and
# 2_build_sample.py uses the downloaded value for BASE_YEAR when it has one.
C2_HIGH_INCOME_THRESHOLD = 9360.0

# Borderline trigger 4 (File 2 §6.2): commercial index providers disagree with
# the C1 assignment. File 2's standing example is South Korea.
INDEX_PROVIDER_DISAGREEMENTS = {
    "KOR": "MSCI classifies South Korea as an emerging market while other "
           "providers treat it as developed",
}


# ==========================================================================
# FILE 3 - OPEN CHOICES FOR THE SUPPLEMENTARY EVENT FLAGS (§5.8)
# ==========================================================================

# The capital-city flags use GeoNames' capital for each country. Where
# GeoNames records none (Israel, for one), those flags stay empty unless you
# name the city here, e.g. {"ISR": "Jerusalem"}. The flags describe events;
# they never enter the TSI.
CAPITAL_OVERRIDES = {}


# ==========================================================================
# §6.2 - OUTCOME VARIABLES (one per hypothesis)
# ==========================================================================

OUTCOMES = {
    "gdp_growth": OutcomeSpec(
        files=("gdp_growth.csv",),
        hypothesis="H1",
        label="GDP growth",
        unit="% per year",
        source="World Bank WDI NY.GDP.MKTP.KD.ZG",
    ),
    # Gross flows both ways (File 1 §3.1; Saeed et al., 2018): terrorism can
    # push outflows up while pulling inflows down, so netting would cancel
    # part of the effect under test.
    "fdi_ratio": OutcomeSpec(
        files=("fdi_inflows.csv", "fdi_outflows.csv"),
        hypothesis="H2",
        label="FDI ratio (inflows + outflows)",
        unit="% of GDP",
        source="World Bank WDI BX.KLT.DINV.WD.GD.ZS + BM.KLT.DINV.WD.GD.ZS",
        # Markets whose FDI ratio AVERAGES more than 100% of GDP over the
        # window are left out of the H2 regressions only: their recorded FDI
        # is mostly financial-centre pass-through capital, not investment in
        # the economy. In this sample that is Malta (195%) and Cyprus (187%);
        # the next highest, the Netherlands, averages 44%. They stay in the sample
        # and in the TSI pool, and §8.5 reports H2 with them as a check.
        # Adopted after the first run flagged them (t6_6_extreme_values.csv) -
        # §6.2 and §8.5 must say so. None switches the rule off.
        exclude_mean_above=100.0,
    ),
}


# ==========================================================================
# §6.4 - CONTROL VARIABLES (the vector X, the same for both outcomes)
# ==========================================================================
# Each is justified by precedent (§6.4). Time-invariant controls in the
# Blomberg-Hess-Orphanides list - the regional dummy and the commodity-
# exporter dummy - are absorbed by the country fixed effects and cannot be
# estimated, so they are not listed; initial income enters in its time-varying
# form, last year's income.
# zero_is_missing=True: an investment rate or trade ratio of exactly 0 is not a
# real value but the World Bank's placeholder for "no data" (Venezuela
# 1998-2011), so it is treated as missing. State this in §6.5.

CONTROLS = {
    "initial_income": ControlSpec(
        file="gdp_per_capita.csv", transform="log", lag=1,
        label="Initial income (log GDP per capita, t-1)",
        unit="log constant 2015 US$",
        precedent="BHO, via Sandler & Enders (2008) §3"),
    "investment_rate": ControlSpec(
        file="investment_rate.csv", zero_is_missing=True,
        label="Investment rate (gross capital formation)", unit="% of GDP",
        precedent="BHO, via Sandler & Enders (2008) §3 - note the mild "
                  "post-treatment risk (§10)"),
    "trade_openness": ControlSpec(
        file="trade_openness.csv", zero_is_missing=True,
        label="Trade openness", unit="% of GDP",
        precedent="BHO panel version; Saeed et al. (2018)"),
    "lagged_growth": ControlSpec(
        file="gdp_per_capita_growth.csv", lag=1,
        label="GDP per capita growth, t-1", unit="% per year",
        precedent="BHO panel version"),
    # ln(1 + inflation/100): follows Saeed et al.'s practice of logging their
    # variables, and keeps hyperinflation years from dominating the estimate.
    "inflation": ControlSpec(
        file="inflation.csv", transform="log_rate",
        label="Inflation, ln(1 + CPI rate)", unit="log points",
        precedent="Saeed et al. (2018)"),
    "market_size": ControlSpec(
        file="gdp_constant_usd.csv", transform="log",
        label="Market size (log real GDP)", unit="log constant 2015 US$",
        precedent="Saeed et al. (2018)"),
    "conflict_internal": ControlSpec(
        file="ucdp_conflict.csv", folder="conflict",
        column="conflict_internal", fill_missing=0.0,
        label="Internal armed conflict (0/1)", unit="0/1",
        precedent="BHO, via Sandler & Enders (2008) §3"),
    "conflict_external": ControlSpec(
        file="ucdp_conflict.csv", folder="conflict",
        column="conflict_external", fill_missing=0.0,
        label="External armed conflict (0/1)", unit="0/1",
        precedent="BHO, via Sandler & Enders (2008) §3"),
}


# ==========================================================================
# §7 - ESTIMATION
# ==========================================================================

# Standard errors (§7.4): "auto" picks Driscoll-Kraay when the Pesaran CD test
# finds cross-sectional dependence and country-clustered errors otherwise.
# "cluster" or "driscoll_kraay" force one type for every regression.
SE_TYPE = "auto"

# §8.7 component decomposition:
#   "separate"  one regression per component, each in place of the TSI
#   "joint"     all four components together in one regression
COMPONENT_ENTRY = "separate"

# §4.5 promises a re-run without the borderline markets (robustness R5); §7.1
# says the four regressions are the whole analysis. Off by default; True adds
# the four regressions on the sample without borderline markets.
ROBUSTNESS_EXCLUDE_BORDERLINE = False


# ==========================================================================
# §6.5 VALIDATION AND OUTPUT
# ==========================================================================

SPOT_CHECK_MARKET_YEARS = 12       # Appendix E: market-years to check by hand
RANDOM_SEED = 20260918

FIGURE_DPI = 300                   # print quality
