"""Central configuration for ChurnGuard AI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTH_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = ROOT / "assets"

# Real dataset (IBM Telco Customer Churn) - must be downloaded by the user.
REAL_DATA_PATH = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
# Synthetic stand-in with the same schema, for testing only.
SYNTH_DATA_PATH = SYNTH_DIR / "telco_synthetic.csv"
CLEAN_DATA_PATH = PROCESSED_DIR / "clean.csv"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
BACKGROUND_PATH = MODELS_DIR / "background.csv"
CURVES_PATH = REPORTS_DIR / "curves.json"

TARGET = "Churn"
ID_COL = "customerID"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_SEARCH_ITER = 10          # random-search iterations per model
DECISION_THRESHOLD = 0.50

# Risk bands on churn probability: < low = Low, < high = Medium, else High.
RISK_THRESHOLDS = {"low": 0.30, "high": 0.60}
