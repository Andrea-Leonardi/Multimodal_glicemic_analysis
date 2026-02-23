from pathlib import Path

# === BASE PROJECT DIR ===
BASE_DIR = Path(__file__).resolve().parents[1]

# === DATA ROOT ===
DATA_DIR = BASE_DIR / "Data"

# === GLOOKO ===
GLOOKO_DIR = DATA_DIR / "glooko"

GLOOKO_EXPORT_1 = GLOOKO_DIR / "16_11_24-13_02_25"
GLOOKO_EXPORT_2 = GLOOKO_DIR / "14_02_25-14_05_25"
GLOOKO_EXPORT_3 = GLOOKO_DIR / "15_05_25-12_08_25"
GLOOKO_EXPORT_4 = GLOOKO_DIR / "13_08_25-10_11_25"
GLOOKO_EXPORT_5 = GLOOKO_DIR / "11_11_25-07_02_26"

# === GLOOKO FILES ===
CGM_PATTERN = "cgm_data_*.csv"
ALARMS_FILE    = "alarms_data_1.csv"

INSULIN_SUBDIR = "Insulin data"
INSULIN_FILE   = "insulin_data_1.csv"
BOLUS_FILE     = "bolus_data_1.csv"


# === XIAOMI ===
XIAOMI_DIR = DATA_DIR / "xiaomi" / "20260208_8284342367_MiFitness_ams1_data_copy"
XIAOMI_FITNESS_FILE = "20260208_8284342367_MiFitness_hlth_center_aggregated_fitness_data.csv"
XIAOMI_TRAINING_FILE = "20260208_8284342367_MiFitness_hlth_center_sport_record.csv"


# === PROCESSED ===

PROCESSED_SUBDIR = DATA_DIR / "processed"

XIAOMI_DATA_CLEANED = PROCESSED_SUBDIR / "xiaomi_data_cleaned.parquet"
GLOOKO_DATA_CLEANED = PROCESSED_SUBDIR / "glooko_data_cleaned.parquet"
DATA_CLEANED = PROCESSED_SUBDIR / "data_cleaned.parquet"
DATA_PROCESSED = PROCESSED_SUBDIR / "data_processed.parquet"