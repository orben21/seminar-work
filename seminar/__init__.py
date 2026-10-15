"""Seminar project: terrorism, GDP growth and FDI in developed and developing
countries - the code behind the action plan and definitional Files 1-3.

Every module works on flat DataFrames keyed by iso3 and year - no MultiIndex.

    specs           declarations of outcome and control series (config.py)
    paths           where every folder and output lives
    countries       country name -> ISO3
    io_tabular      read any csv/xlsx of country-year data
    sources         download and parse WDI, UNDP HDR, OGHIST, UCDP, GeoNames
    terror          File 3 Part A: K1-K6, the floor, traits, flags, aggregation
    qualification   File 1: market qualification Q1-Q6
    classification  File 2: developed / developing, C1-C3, borderline flags
    tsi             File 3 Part B: the Terrorism Severity Index
    panel           §6: the country-year panel
    descriptives    §6.6: descriptive statistics
    diagnostics     §7.4: diagnostic tests and the standard-error choice
    regression      §7: the four regressions, build-up, decomposition, tests
    viz             the data figures of the plan's master list
    report          CSV tables and the HTML report
    sample_data     a planted effect on the real panel, to test the regressions
"""

__version__ = "2.0"
