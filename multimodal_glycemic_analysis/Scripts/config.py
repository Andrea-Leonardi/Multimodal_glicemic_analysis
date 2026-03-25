from __future__ import annotations

from pathlib import Path

# === BASE PROJECT DIR ===
BASE_DIR = Path(__file__).resolve().parents[1]

# === DATA ROOT ===
DATA_DIR = BASE_DIR / "Data"
TIMEZONE = "Europe/Rome"

# === GLOOKO ===
GLOOKO_DIR = DATA_DIR / "glooko"
CGM_PATTERN = "cgm_data_*.csv"
ALARMS_FILE = "alarms_data_1.csv"
INSULIN_SUBDIR = "Insulin data"
INSULIN_FILE = "insulin_data_1.csv"
BOLUS_FILE = "bolus_data_1.csv"

# === XIAOMI ===
XIAOMI_DIR = DATA_DIR / "xiaomi" / "20260208_8284342367_MiFitness_ams1_data_copy"
XIAOMI_FITNESS_FILE = "20260208_8284342367_MiFitness_hlth_center_aggregated_fitness_data.csv"
XIAOMI_TRAINING_FILE = "20260208_8284342367_MiFitness_hlth_center_sport_record.csv"

# === PROCESSED ===
PROCESSED_SUBDIR = DATA_DIR / "processed"
XIAOMI_DATA_CLEANED = PROCESSED_SUBDIR / "xiaomi_data_cleaned.parquet"
GLOOKO_DATA_CLEANED = PROCESSED_SUBDIR / "glooko_data_cleaned.parquet"
DATA_CLEANED = PROCESSED_SUBDIR / "data_cleaned.parquet"
DATA_LAGGED = PROCESSED_SUBDIR / "data_lagged.parquet"
DATA_PROCESSED = PROCESSED_SUBDIR / "data_processed.parquet"
DATA_PROCESSED_CSV = PROCESSED_SUBDIR / "dataproc.csv"
ML_RESULTS = PROCESSED_SUBDIR / "ml_results.csv"

# === MODELING DEFAULTS ===
DEFAULT_TARGET_COLUMN = "bg_auc_above_limit_rate"
DEFAULT_LAG_DAYS = 2
MIN_FEATURE_AVAILABILITY = 0.75
DEFAULT_LAG_COLUMNS = [
    "training",
    "bg_mean",
    "bg_sd",
    "bg_max",
    DEFAULT_TARGET_COLUMN,
    "cartridge_loaded",
    "carb_daily",
    "sl_tot_duration",
    "sl_score",
    "sl_bedtime_mfm",
    "steps",
]


def ensure_processed_dir() -> Path:
    PROCESSED_SUBDIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_SUBDIR


def resolve_glooko_exports() -> list[Path]:
    if not GLOOKO_DIR.exists():
        return []
    return sorted(path for path in GLOOKO_DIR.iterdir() if path.is_dir())


def require_existing_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")
    return path
