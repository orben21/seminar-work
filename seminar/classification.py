"""File 2: is an admitted market developed or developing?

Three criteria, applied in a fixed order (File 2 §2):

    C1  HDI >= 0.800 -> developed        primary and decisive (Ahmad et al.,
                                          2022; Dasic et al., 2020)
    C2  World Bank income group           confirms or contradicts C1 (Fantom &
                                          Serajuddin, 2016): high income
                                          confirms developed
    C3  UNDP / World Bank / IMF agree     at least two of three must agree with
                                          C1 (Nielsen, 2011)

Every market is classified ONCE, at the base year of the window, and keeps its
group for the whole panel (File 2 §6.1). A market is flagged borderline if its
HDI is within 0.02 of 0.800, if C2 contradicts C1, if fewer than two of the
three institutions agree with C1, or if commercial index providers disagree
(File 2 §6.2). Flagged markets KEEP their C1 group; the flag is a reporting
obligation, not a reclassification.

The thresholds above are File 2's definitions and are constants here. The HDI
values come from UNDP's composite time series, which is recomputed on the
current (post-2010) methodology for every year, so a 1998 value may be compared
with today's 0.800 line (File 2 §3.4 warns against mixing vintages).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import countries

HDI_THRESHOLD = 0.800          # C1, File 2 §3.1
BORDERLINE_BAND = 0.02         # File 2 §6.2
UNDP_QUANTILE = 0.75           # C3: UNDP's developed group is the top quartile
                               # of the HDI distribution (File 2 §5; Nielsen)
HDI_TIERS = ((0.800, "very high"), (0.700, "high"), (0.550, "medium"),
             (-np.inf, "low"))  # current UNDP four-tier scheme (File 2 §3.4)
INCOME_GROUP_NAMES = {"L": "low income", "LM": "lower-middle income",
                      "UM": "upper-middle income", "H": "high income",
                      "below-H": "below the high-income line"}

DEVELOPED, DEVELOPING = "developed", "developing"


@dataclass
class ClassificationRules:
    """The open choices File 2 leaves to the study, read from config.py."""

    base_year: int
    window: Tuple[int, int]
    c2_source: str = "history"              # "history" | "gni"
    high_income_threshold: float = 9360.0   # GNI Atlas line of the base year
    index_provider_disagreements: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.c2_source not in ("history", "gni"):
            raise ValueError("C2_SOURCE must be 'history' or 'gni'")


def hdi_tier(value: float) -> str:
    if not np.isfinite(value):
        return ""
    for floor, name in HDI_TIERS:
        if value >= floor:
            return name
    return ""


def value_at(table: pd.DataFrame, column: str, iso3: str, base_year: int,
             window: Tuple[int, int]) -> Tuple[float, Optional[int]]:
    """The value at the base year, or failing that the earliest year of the
    window that has one. Returns (value, year used)."""
    own = table[(table["iso3"] == iso3) & table[column].notna()]
    exact = own[own["year"] == base_year]
    if not exact.empty:
        return float(exact[column].iloc[0]), base_year
    inside = own[own["year"].between(*window)].sort_values("year")
    if not inside.empty:
        return float(inside[column].iloc[0]), int(inside["year"].iloc[0])
    return np.nan, None


def _as_group(developed: Optional[bool]) -> str:
    if developed is None:
        return "not classified"
    return DEVELOPED if developed else DEVELOPING


def classify_markets(markets: Sequence[str],
                     hdi: pd.DataFrame,
                     gni_atlas: pd.DataFrame,
                     income_history: Optional[pd.DataFrame],
                     imf_advanced: pd.DataFrame,
                     rules: ClassificationRules,
                     descriptors: Optional[Dict[str, pd.DataFrame]] = None
                     ) -> pd.DataFrame:
    """The §4.7 master table: one row per admitted market.

    hdi              DataFrame[iso3, year, hdi] - every country, so the UNDP
                     top-quartile line is computed on the whole distribution
    gni_atlas        DataFrame[iso3, year, gni_atlas]
    income_history   DataFrame[iso3, year, income_group] with L/LM/UM/H - the
                     World Bank's own historical assignment (OGHIST), or None
    imf_advanced     DataFrame[iso3, advanced_since]
    descriptors      optional {name: DataFrame[iso3, year, name]} recorded at
                     the base year but never used to assign a group (§7)
    """
    base = rules.base_year
    base_hdi = hdi.loc[hdi["year"] == base, "hdi"].dropna()
    undp_line = (float(base_hdi.quantile(UNDP_QUANTILE))
                 if len(base_hdi) else np.nan)
    imf_since = dict(zip(imf_advanced["iso3"],
                         imf_advanced["advanced_since"].astype(int)))

    rows = []
    for iso3 in sorted(markets):
        row = {"iso3": iso3, "country": countries.name_of(iso3)}

        # ---- C1 HDI ---------------------------------------------------------
        value, year_used = value_at(hdi, "hdi", iso3, base, rules.window)
        row["hdi"] = value
        row["hdi_year"] = year_used
        row["hdi_tier"] = hdi_tier(value)
        c1_developed = (None if not np.isfinite(value)
                        else round(value, 3) >= HDI_THRESHOLD)
        row["c1_group"] = _as_group(c1_developed) if c1_developed is not None \
            else "unclassifiable"

        # ---- C2 World Bank income group --------------------------------------
        gni, gni_year = value_at(gni_atlas, "gni_atlas", iso3, base,
                                 rules.window)
        row["gni_atlas"] = gni
        row["gni_year"] = gni_year
        group_code, source = None, ""
        if rules.c2_source == "history" and income_history is not None:
            match = income_history[(income_history["iso3"] == iso3)
                                   & (income_history["year"] == base)]
            if not match.empty and isinstance(match["income_group"].iloc[0],
                                              str):
                group_code = match["income_group"].iloc[0]
                source = f"World Bank classification of {base} (OGHIST)"
        if group_code is None and np.isfinite(gni):
            group_code = ("H" if gni > rules.high_income_threshold
                          else "below-H")
            source = (f"GNI Atlas {gni_year} vs "
                      f"{rules.high_income_threshold:,.0f} line")
        row["income_group"] = INCOME_GROUP_NAMES.get(group_code, group_code
                                                     or "")
        row["c2_source"] = source
        c2_developed = None if group_code is None else group_code == "H"
        row["c2_group"] = _as_group(c2_developed)

        # ---- C3 three institutions ---------------------------------------------
        imf_dev = iso3 in imf_since and imf_since[iso3] <= base
        undp_dev = (None if not (np.isfinite(value) and np.isfinite(undp_line))
                    else value >= undp_line)
        wb_dev = c2_developed
        row["imf_group"] = _as_group(imf_dev)
        row["undp_group"] = _as_group(undp_dev)
        row["wb_group"] = _as_group(wb_dev)
        if c1_developed is None:
            row["c3_agreements"] = np.nan
        else:
            row["c3_agreements"] = sum(
                1 for vote in (imf_dev, undp_dev, wb_dev)
                if vote is not None and vote == c1_developed)

        # ---- borderline triggers (File 2 §6.2) ---------------------------------
        triggers = []
        if c1_developed is not None:
            if abs(round(value, 3) - HDI_THRESHOLD) <= BORDERLINE_BAND + 1e-9:
                triggers.append(f"HDI {value:.3f} within {BORDERLINE_BAND} "
                                f"of {HDI_THRESHOLD:.3f}")
            if c2_developed is not None and c2_developed != c1_developed:
                triggers.append(f"C2 contradicts C1 ({row['income_group']})")
            if row["c3_agreements"] < 2:
                triggers.append(f"only {int(row['c3_agreements'])} of 3 "
                                f"institutions agree")
            if iso3 in rules.index_provider_disagreements:
                triggers.append(rules.index_provider_disagreements[iso3])
        row["borderline"] = bool(triggers)
        row["borderline_triggers"] = "; ".join(triggers)

        row["group"] = row["c1_group"] if c1_developed is not None else np.nan
        row["developed"] = (np.nan if c1_developed is None
                            else float(c1_developed))

        for name, table in (descriptors or {}).items():
            row[name], _ = value_at(table, name, iso3, base, rules.window)
        rows.append(row)

    table = pd.DataFrame(rows)
    order = {DEVELOPED: 0, DEVELOPING: 1}
    table["_order"] = table["group"].map(order).fillna(2)
    table = (table.sort_values(["_order", "hdi"], ascending=[True, False])
             .drop(columns="_order").reset_index(drop=True))
    table.attrs["undp_line"] = undp_line
    return table


def summarise(table: pd.DataFrame, base_year: int,
              max_names: int = 14) -> str:
    """Readable summary of the split, printed by 2_build_sample.py."""
    lines = [f"UNDP top-quartile line at the base year: "
             f"HDI {table.attrs.get('undp_line', float('nan')):.3f}"]
    for group in (DEVELOPED, DEVELOPING):
        members = table.loc[table["group"] == group, "iso3"].tolist()
        shown = ", ".join(members[:max_names])
        more = (f", +{len(members) - max_names} more"
                if len(members) > max_names else "")
        lines.append(f"  {group:<11} {len(members):>3} markets: {shown}{more}")
    flagged = table[table["borderline"]]
    lines.append(f"  borderline  {len(flagged):>3} markets: "
                 + ", ".join(flagged["iso3"]))
    shifted = table[table["hdi_year"].notna()
                    & (table["hdi_year"] != base_year)]
    for _, row in shifted.iterrows():
        lines.append(f"  note: {row['iso3']} has no HDI at the base year; "
                     f"classified on its {int(row['hdi_year'])} value "
                     f"({row['hdi']:.3f}) - state this in §4")
    unclassifiable = table[table["group"].isna()]
    if len(unclassifiable):
        lines.append(f"  NO HDI      {len(unclassifiable):>3} markets: "
                     + ", ".join(unclassifiable["iso3"])
                     + "  (dropped: C1 cannot assign them)")
    return "\n".join(lines)
