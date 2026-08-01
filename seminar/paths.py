"""Every folder the project uses, derived from the location of this package."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"      # dependent variables
CONTROLS_DIR = DATA_DIR / "controls"    # independent controls
FILTERS_DIR = DATA_DIR / "filters"      # classification / filtration data
REFERENCE_DIR = DATA_DIR / "reference"  # country_aliases.csv
SAMPLE_DIR = DATA_DIR / "sample"        # synthetic data for testing the pipeline
CACHE_DIR = DATA_DIR / "cache"          # generated parquet caches

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_HTML = OUTPUT_DIR / "report.html"

GTD_EVENTS_PARQUET = CACHE_DIR / "gtd_events.parquet"
GTD_COUNTRY_YEAR_PARQUET = CACHE_DIR / "gtd_country_year.parquet"
COUNTRY_ALIASES_CSV = REFERENCE_DIR / "country_aliases.csv"
UNMATCHED_COUNTRIES_CSV = OUTPUT_DIR / "unmatched_countries.csv"

ALL_DIRS = (
    METRICS_DIR, CONTROLS_DIR, FILTERS_DIR, REFERENCE_DIR, SAMPLE_DIR,
    CACHE_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR,
)


def ensure_dirs():
    """Create every project folder if it does not exist yet."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
