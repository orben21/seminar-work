"""Every folder the project uses, derived from the location of this package."""

from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
WORLDBANK_DIR = DATA_DIR / "worldbank"            # WDI series (outcomes, controls)
UNDP_DIR = DATA_DIR / "undp"                      # HDI and its three dimensions
CLASSIFICATION_DIR = DATA_DIR / "classification"  # WB income-group history, IMF list
CONFLICT_DIR = DATA_DIR / "conflict"              # UCDP/PRIO conflict indicators
REFERENCE_DIR = DATA_DIR / "reference"            # country aliases, cities
CACHE_DIR = DATA_DIR / "cache"                    # generated parquet cache

OUTPUT_DIR = PROJECT_ROOT / "outputs"
STAGE1_DIR = OUTPUT_DIR / "stage1"                # the files Part 5 says to upload
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_HTML = OUTPUT_DIR / "report.html"
# 3_run_analysis.py --sample writes here, never over the real results.
SAMPLE_CHECK_DIR = OUTPUT_DIR / "sample_check"

GTD_EVENTS_PARQUET = CACHE_DIR / "gtd_events.parquet"
COUNTRY_ALIASES_CSV = REFERENCE_DIR / "country_aliases.csv"
CITIES_CSV = REFERENCE_DIR / "cities.csv"
UNMATCHED_COUNTRIES_CSV = OUTPUT_DIR / "unmatched_countries.csv"

# Stage 1 outputs that do not depend on the window.
SAMPLE_MARKETS_CSV = STAGE1_DIR / "sample_markets.csv"

ALL_DIRS = (
    WORLDBANK_DIR, UNDP_DIR, CLASSIFICATION_DIR, CONFLICT_DIR, REFERENCE_DIR,
    CACHE_DIR, OUTPUT_DIR, STAGE1_DIR, FIGURES_DIR, TABLES_DIR,
)


def qualifying_events_csv(window: Tuple[int, int]) -> Path:
    """Part 5, Stage 1 export 1: event level, sample countries only."""
    return STAGE1_DIR / f"gtd_qualifying_events_{window[0]}_{window[1]}.csv"


def country_year_csv(window: Tuple[int, int]) -> Path:
    """Part 5, Stage 1 export 2: one row per market-year."""
    return STAGE1_DIR / f"gtd_country_year_{window[0]}_{window[1]}.csv"


def ensure_dirs():
    """Create every project folder if it does not exist yet."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
