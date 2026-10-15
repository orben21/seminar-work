# -*- coding: utf-8 -*-
"""Tests for the file reader - the part everything else depends on.

Run from PyCharm (green arrow) or:  python -m tests.test_io_tabular
No test framework needed; it prints PASS/FAIL and exits non-zero on failure.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from seminar import io_tabular
from seminar.specs import ControlSpec

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}"
          + (f"   {detail}" if detail and not condition else ""))


def write(directory, name, text):
    path = Path(directory) / name
    path.write_text(text, encoding="utf-8")
    return path


def test_long_and_wide_agree(directory):
    """The same numbers in either layout must produce the same table."""
    long_path = write(directory, "long.csv",
                      "country,year,value\n"
                      "United States,1990,10.5\n"
                      "United States,1991,11.5\n"
                      "Germany,1990,20.5\n"
                      "Germany,1991,21.5\n")
    wide_path = write(directory, "wide.csv",
                      "country,1990,1991\n"
                      "United States,10.5,11.5\n"
                      "Germany,20.5,21.5\n")
    long_table = io_tabular.load_table(long_path, "x")
    wide_table = io_tabular.load_table(wide_path, "x")

    check("long and wide give identical tables",
          long_table.equals(wide_table),
          f"\n{long_table}\n vs \n{wide_table}")
    check("columns are exactly iso3, year, value",
          list(long_table.columns) == ["iso3", "year", "x"])
    check("index is a plain RangeIndex, not a MultiIndex",
          isinstance(long_table.index, pd.RangeIndex))
    check("country names resolved to ISO3",
          sorted(long_table["iso3"].unique()) == ["DEU", "USA"])


def test_iso3_input_matches_names(directory):
    """A file keyed by ISO3 must load the same as one keyed by name."""
    by_name = write(directory, "byname.csv",
                    "country,year,value\nUnited States,1990,10.5\n")
    by_code = write(directory, "bycode.csv",
                    "iso3,year,value\nUSA,1990,10.5\n")
    check("ISO3 codes and country names agree",
          io_tabular.load_table(by_name, "x").equals(
              io_tabular.load_table(by_code, "x")))


def test_messy_values(directory):
    """Thousands separators, '..' for missing, and a metadata preamble."""
    path = write(directory, "messy.csv",
                 "World Development Indicators\n"
                 "Last updated: whenever\n"
                 "country,year,value\n"
                 "United States,1990,\"1,234.5\"\n"
                 "United States,1991,..\n"
                 "Germany,1990,2000\n")
    table = io_tabular.load_table(path, "x")
    check("preamble rows skipped and header found", len(table) == 2)
    check("thousands separator parsed",
          float(table.loc[table["iso3"] == "USA", "x"].iloc[0]) == 1234.5)
    check("'..' treated as missing and dropped",
          1991 not in set(table["year"]))


def test_unknown_country_is_reported(directory):
    path = write(directory, "unknown.csv",
                 "country,year,value\n"
                 "United States,1990,1\n"
                 "Neverland,1990,2\n")
    table = io_tabular.load_table(path, "x")
    check("unrecognised country dropped, real one kept",
          list(table["iso3"]) == ["USA"])
    check("unrecognised name recorded for the report",
          any("Neverland" in names
              for names in io_tabular.countries.unmatched().values()))


def test_clear_errors(directory):
    missing = Path(directory) / "nope.csv"
    try:
        io_tabular.load_table(missing, "x")
        check("missing file raises a clear error", False)
    except io_tabular.DataFileError as error:
        check("missing file raises a clear error", "not found" in str(error))

    path = write(directory, "badcol.csv", "country,year,value\nUSA,1990,1\n")
    try:
        io_tabular.load_table(path, "x", ControlSpec(file="badcol.csv",
                                                     column="wrong_name"))
        check("wrong column name raises a clear error", False)
    except io_tabular.DataFileError as error:
        check("wrong column name raises a clear error",
              "wrong_name" in str(error) and "Available columns" in str(error))


def test_transforms(directory):
    path = write(directory, "t.csv",
                 "country,year,value\nUSA,1990,100\nUSA,1991,110\n")
    table = io_tabular.load_table(path, "x")
    changed = io_tabular.apply_transform(table.copy(), "x", "pct_change")
    check("pct_change computed per country",
          len(changed) == 1 and abs(float(changed["x"].iloc[0]) - 10.0) < 1e-9)
    logged = io_tabular.apply_transform(table.copy(), "x", "log")
    check("log transform applied",
          abs(float(logged["x"].iloc[0]) - 4.605170) < 1e-5)
    rate = write(directory, "rate.csv",
                 "country,year,value\nUSA,1990,100\nUSA,1991,-100\n")
    rated = io_tabular.apply_transform(io_tabular.load_table(rate, "x"), "x",
                                       "log_rate")
    check("log_rate is ln(1 + x/100) and drops -100%",
          len(rated) == 1 and abs(float(rated["x"].iloc[0])
                                  - 0.693147) < 1e-5)


def main() -> int:
    print("Testing the data loading layer\n")
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        test_long_and_wide_agree(directory)
        test_iso3_input_matches_names(directory)
        test_messy_values(directory)
        test_unknown_country_is_reported(directory)
        test_clear_errors(directory)
        test_transforms(directory)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
