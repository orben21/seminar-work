# -*- coding: utf-8 -*-
"""STEP 1 - convert the Global Terrorism Database into a fast cache.

Run this once. It reads the 104 MB GTD xlsx, keeps only the fields File 3 and
plan §5.2-§5.8 name (qualification, severity traits, event flags), and writes
data/cache/gtd_events.parquet. Every later run loads that in under a second.

Takes a few minutes. Run it again only if you get a newer GTD file, or if
2_build_sample.py says the cache predates the current code.
"""

import sys

import config
from seminar import countries, paths, terror


def main() -> int:
    paths.ensure_dirs()
    print("=" * 72)
    print("STEP 1  building the GTD cache")
    print("=" * 72)

    try:
        events = terror.convert_to_parquet(config.GTD_XLSX,
                                           sheet_name=config.GTD_SHEET)
    except terror.GTDError as error:
        print(f"\nERROR: {error}")
        return 1

    print("\nSanity checks")
    print(terror.describe(events, config.WINDOW))
    funnel = terror.event_funnel(events, config.WINDOW, "all GTD countries")
    print(f"\nFile 3 qualification funnel, {config.WINDOW[0]}-"
          f"{config.WINDOW[1]} (all countries)")
    for _, row in funnel.iterrows():
        print(f"  {row['label']:<48}{int(row['events']):>9,}  "
              f"({row['pct_of_window']:.1f}%)")

    report = countries.write_unmatched_report()
    if report:
        print(f"\nUnresolved country names listed in {report}")

    print("\nDone. Next: 2_build_sample.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
