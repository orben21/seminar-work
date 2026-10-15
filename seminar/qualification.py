"""File 1: which markets enter the sample (rules Q1-Q6).

A market is the national economy of a sovereign statistical unit, observed
annually (File 1 §2). A candidate enters the sample only if it passes every
rule; Q1-Q4 are hard filters and Q5-Q6 exclusions applied after them (§4):

    Q1  statistical identity      one consistently named, sovereign unit in
                                  both the GTD and World Bank accounts over the
                                  whole window
    Q2  observable outcomes       GDP growth and FDI inflows + outflows for
                                  every year of the window (balanced panel)
    Q3  demonstrated openness     a non-trivial trade ratio and FDI ratio
    Q4  presence in terror data   at least one qualifying event (File 3)
    Q5  data-integrity exclusion  war / contested statehood (listed, reasoned)
    Q6  degenerate-case exclusion too few qualifying events for any
                                  within-country variation

Every rule is evaluated for every candidate, so the audit shows all the
reasons a country fails, and the exclusion log records the first rule it
failed and what triggered it (File 1 §4.1: "recorded, not silently applied").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import countries

RULES = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
RULE_NAMES = {
    "Q1": "statistical identity",
    "Q2": "observable outcome variables",
    "Q3": "demonstrated openness",
    "Q4": "presence in the terror data",
    "Q5": "data-integrity exclusion",
    "Q6": "degenerate-case exclusion",
}


@dataclass
class QualificationRules:
    """The open choices File 1 leaves to the study, read from config.py."""

    window: Tuple[int, int]
    non_sovereign: Dict[str, str] = field(default_factory=dict)
    territorial_changes: Dict[str, str] = field(default_factory=dict)
    min_trade_ratio: float = 10.0
    min_fdi_ratio: float = 0.5
    exclusions: Dict[str, str] = field(default_factory=dict)
    min_active_years: int = 3
    candidates: Optional[Sequence[str]] = None


@dataclass
class QualificationResult:
    audit: pd.DataFrame          # one row per candidate, every rule
    exclusion_log: pd.DataFrame  # Appendix C: country, rule invoked, trigger
    funnel: pd.DataFrame         # Figure 6
    admitted: List[str]          # ISO3 codes of the sample


def _year_ranges(years: Iterable[int]) -> str:
    """[1998, 1999, 2000, 2004] -> '1998-2000, 2004'."""
    years = sorted(set(int(y) for y in years))
    if not years:
        return ""
    spans, start, previous = [], years[0], years[0]
    for year in years[1:]:
        if year == previous + 1:
            previous = year
            continue
        spans.append(f"{start}-{previous}" if start != previous else f"{start}")
        start = previous = year
    spans.append(f"{start}-{previous}" if start != previous else f"{start}")
    return ", ".join(spans)


def _missing_years(table: pd.DataFrame, iso3: str, column: str,
                   years: List[int]) -> List[int]:
    have = set(table.loc[(table["iso3"] == iso3) & table[column].notna(),
                         "year"].astype(int))
    return [y for y in years if y not in have]


def qualify_markets(events: pd.DataFrame,
                    qualifying: pd.DataFrame,
                    gdp_growth: pd.DataFrame,
                    fdi_inflows: pd.DataFrame,
                    fdi_outflows: pd.DataFrame,
                    trade: pd.DataFrame,
                    rules: QualificationRules) -> QualificationResult:
    """Apply Q1-Q6 to every candidate.

    events       the whole GTD cache (for the candidate list and Q1 naming)
    qualifying   events passing File 3's filters and floor (Q4, Q6)
    gdp_growth, fdi_inflows, fdi_outflows, trade
                 World Bank series as DataFrame[iso3, year, <name>]
    """
    first, last = rules.window
    years = list(range(first, last + 1))
    in_window = events[events["iyear"].between(first, last)]

    # ---- the candidate list: every country the GTD records in the window --
    names = (in_window.groupby("country_txt", dropna=True)
             .agg(iso3=("iso3", "first"), gtd_events=("eventid", "count"))
             .reset_index())
    if rules.candidates:
        wanted = {str(c).strip().upper() for c in rules.candidates}
        names = names[names["iso3"].isin(wanted)]
        absent = sorted(wanted - set(names["iso3"].dropna()))
        names = pd.concat([names, pd.DataFrame(
            {"country_txt": [countries.name_of(c) for c in absent],
             "iso3": absent, "gtd_events": 0})], ignore_index=True)
    names_per_iso3 = (names.dropna(subset=["iso3"]).groupby("iso3")
                      ["country_txt"].apply(lambda s: sorted(s)).to_dict())

    wb_countries = set(gdp_growth["iso3"])
    fdi = fdi_inflows.merge(fdi_outflows, on=["iso3", "year"], how="outer")
    fdi["fdi_ratio"] = fdi["fdi_inflows"] + fdi["fdi_outflows"]
    window_trade = trade[trade["year"].between(first, last)]
    window_fdi = fdi[fdi["year"].between(first, last)]

    per_country = (qualifying.groupby("iso3")
                   .agg(qualifying_events=("eventid", "count"),
                        active_years=("iyear", "nunique")))

    rows = []
    for _, candidate in names.iterrows():
        name = candidate["country_txt"]
        iso3 = candidate["iso3"] if isinstance(candidate["iso3"], str) else None
        row = {"gtd_name": name, "iso3": iso3,
               "country": countries.name_of(iso3) if iso3 else name,
               "gtd_events_window": int(candidate["gtd_events"])}
        detail: Dict[str, str] = {}

        # ---- Q1 statistical identity --------------------------------------
        if iso3 is None:
            detail["Q1"] = f"GTD location '{name}' is not a country"
        elif iso3 in rules.non_sovereign:
            detail["Q1"] = (f"not a sovereign statistical unit: "
                            f"{rules.non_sovereign[iso3]}")
        elif len(names_per_iso3.get(iso3, [])) > 1:
            detail["Q1"] = ("named inconsistently in the GTD within the "
                            "window: " + " / ".join(names_per_iso3[iso3]))
        elif iso3 in rules.territorial_changes:
            detail["Q1"] = (f"not the same unit over the whole window: "
                            f"{rules.territorial_changes[iso3]}")
        elif iso3 not in wb_countries:
            detail["Q1"] = "not in World Bank national accounts"

        # ---- Q2 observable outcomes (balanced) ----------------------------
        if iso3 is not None:
            gaps, any_missing = [], set()
            for label, table, column in (
                    ("GDP growth", gdp_growth, "gdp_growth"),
                    ("FDI inflows", fdi_inflows, "fdi_inflows"),
                    ("FDI outflows", fdi_outflows, "fdi_outflows")):
                missing = _missing_years(table, iso3, column, years)
                any_missing |= set(missing)
                if missing:
                    gaps.append(f"{label} missing {_year_ranges(missing)}")
            row["q2_missing_years"] = len(any_missing)
            if gaps:
                detail["Q2"] = "; ".join(gaps)
        else:
            row["q2_missing_years"] = len(years)
            detail["Q2"] = "no country to look up"

        # ---- Q3 demonstrated openness -------------------------------------
        trade_mean = (window_trade.loc[window_trade["iso3"] == iso3,
                                       "trade_openness"].mean()
                      if iso3 else np.nan)
        fdi_mean = (window_fdi.loc[window_fdi["iso3"] == iso3,
                                   "fdi_ratio"].mean() if iso3 else np.nan)
        row["trade_ratio_mean"] = trade_mean
        row["fdi_ratio_mean"] = fdi_mean
        problems = []
        if not np.isfinite(trade_mean):
            problems.append("no trade ratio data")
        elif trade_mean < rules.min_trade_ratio:
            problems.append(f"trade ratio averages {trade_mean:.1f}% of GDP "
                            f"(< {rules.min_trade_ratio:g}%)")
        if not np.isfinite(fdi_mean):
            problems.append("no FDI ratio data")
        elif fdi_mean < rules.min_fdi_ratio:
            problems.append(f"FDI ratio averages {fdi_mean:.2f}% of GDP "
                            f"(< {rules.min_fdi_ratio:g}%)")
        if problems:
            detail["Q3"] = "; ".join(problems)

        # ---- Q4 presence in the terror data -------------------------------
        stats = (per_country.loc[iso3] if iso3 in per_country.index
                 else pd.Series({"qualifying_events": 0, "active_years": 0}))
        row["qualifying_events"] = int(stats["qualifying_events"])
        row["active_years"] = int(stats["active_years"])
        if row["qualifying_events"] == 0:
            detail["Q4"] = (f"no qualifying event {first}-{last} "
                            f"({row['gtd_events_window']} GTD events before "
                            f"the File 3 filters)")

        # ---- Q5 data-integrity exclusion ----------------------------------
        if iso3 in rules.exclusions:
            detail["Q5"] = rules.exclusions[iso3]

        # ---- Q6 degenerate case -------------------------------------------
        if 0 < row["active_years"] < rules.min_active_years:
            detail["Q6"] = (f"qualifying events in only {row['active_years']} "
                            f"of {len(years)} years "
                            f"({row['qualifying_events']} events; minimum "
                            f"{rules.min_active_years} years)")

        for rule in RULES:
            row[f"{rule}_pass"] = rule not in detail
            row[f"{rule}_detail"] = detail.get(rule, "")
        failed = [rule for rule in RULES if rule in detail]
        row["excluded_at"] = failed[0] if failed else ""
        row["trigger"] = detail[failed[0]] if failed else ""
        row["also_fails"] = ", ".join(failed[1:])
        row["admitted"] = not failed
        rows.append(row)

    audit = pd.DataFrame(rows).sort_values(["admitted", "excluded_at",
                                            "country"],
                                           ascending=[False, True, True])
    audit = audit.reset_index(drop=True)

    excluded = audit[~audit["admitted"]]
    exclusion_log = pd.DataFrame({
        "country": excluded["country"], "iso3": excluded["iso3"],
        "gtd_name": excluded["gtd_name"],
        "rule_invoked": excluded["excluded_at"],
        "rule_name": excluded["excluded_at"].map(RULE_NAMES),
        "trigger": excluded["trigger"],
        "also_fails": excluded["also_fails"],
    }).reset_index(drop=True)

    start_label = ("Candidate list (config)" if rules.candidates
                   else f"All countries in the GTD, {first}-{last}")
    funnel_rows = [{"stage": "start", "label": start_label,
                    "remaining": len(audit), "dropped": 0}]
    remaining = audit
    for rule in RULES:
        survivors = remaining[remaining[f"{rule}_pass"]]
        funnel_rows.append({"stage": rule,
                            "label": f"{rule} {RULE_NAMES[rule]}",
                            "remaining": len(survivors),
                            "dropped": len(remaining) - len(survivors)})
        remaining = survivors
    funnel = pd.DataFrame(funnel_rows)

    admitted = sorted(audit.loc[audit["admitted"], "iso3"].tolist())
    return QualificationResult(audit=audit, exclusion_log=exclusion_log,
                               funnel=funnel, admitted=admitted)
