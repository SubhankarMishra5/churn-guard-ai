"""Dataset discovery, loading and schema validation."""
from __future__ import annotations

import pandas as pd

from config import settings
from config.schema import RAW_FEATURES

SETUP_HELP = (
    "No dataset found.\n"
    "  Option A (real data): download the IBM Telco Customer Churn CSV and save it as\n"
    f"      {settings.REAL_DATA_PATH}\n"
    "  Option B (testing only): run  python -m src.data_generation  to create a\n"
    "      clearly-labelled SYNTHETIC dataset, or train with  --generate-synthetic."
)


class DatasetNotFoundError(FileNotFoundError):
    pass


class SchemaError(ValueError):
    pass


def resolve_data_path(path=None):
    """Return (path, source) where source is 'real', 'synthetic' or 'custom'."""
    if path is not None:
        return path, "custom"
    if settings.REAL_DATA_PATH.exists():
        return settings.REAL_DATA_PATH, "real"
    if settings.SYNTH_DATA_PATH.exists():
        return settings.SYNTH_DATA_PATH, "synthetic"
    raise DatasetNotFoundError(SETUP_HELP)


def validate_raw_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    need = list(RAW_FEATURES) + ([settings.TARGET] if require_target else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SchemaError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise SchemaError("Dataset has no rows.")


def load_raw_data(path=None):
    """Load and schema-check the raw CSV. Returns (DataFrame, source)."""
    p, source = resolve_data_path(path)
    df = pd.read_csv(p)
    validate_raw_schema(df)
    return df, source
