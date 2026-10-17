from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATAMART_DIR = PROJECT_ROOT / "datamart"

SOURCE_FILES = {
    "features_attributes": DATA_DIR / "features_attributes.csv",
    "features_financials": DATA_DIR / "features_financials.csv",
    "feature_clickstream": DATA_DIR / "feature_clickstream.csv",
    "lms_loan_daily": DATA_DIR / "lms_loan_daily.csv",
}

BRONZE_DIR = DATAMART_DIR / "bronze"
SILVER_DIR = DATAMART_DIR / "silver"
GOLD_DIR = DATAMART_DIR / "gold"
