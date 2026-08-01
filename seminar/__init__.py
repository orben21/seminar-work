"""Seminar project: the effect of terrorism on long-term country metrics.

Every module works on the same flat DataFrame shape - columns iso3, year and
one value column - so nothing here ever needs a MultiIndex.

    specs       what you declare in config.py
    paths       where every folder lives
    countries   country name -> ISO3
    io_tabular  read any csv/xlsx of country-year data
    terror      the Global Terrorism Database side
    filters     developed vs developing tagging
    panel       assemble the table for one metric
    regression  the nested models A/B/C/D
    viz         figures
    report      CSV tables and the HTML report
    sample_data synthetic data with a known effect, to test the pipeline
"""

__version__ = "1.0"
