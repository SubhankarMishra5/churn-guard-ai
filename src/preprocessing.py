"""Cleaning, validation, outlier analysis and the sklearn preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import settings
from config.schema import (ALLOWED, CATEGORICAL_RAW, MODEL_CATEGORICAL,
                           MODEL_NUMERIC, NUMERIC_RAW, RANGES, RAW_FEATURES)


class InputValidationError(ValueError):
    """Raised with a list of human-readable problems."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# ----------------------------------------------------------------- cleaning
def clean_data(df: pd.DataFrame):
    """Clean the raw training data. Returns (clean_df, report dict)."""
    report = {"rows_raw": int(len(df))}
    d = df.copy()

    for c in d.columns:
        if pd.api.types.is_string_dtype(d[c]) or d[c].dtype == object:
            d[c] = d[c].astype("string").str.strip()

    report["duplicates_removed"] = int(d.duplicated().sum())
    d = d.drop_duplicates().reset_index(drop=True)
    # Same feature values under different IDs are legitimate -> reported, not dropped.
    feat_cols = [c for c in d.columns if c != settings.ID_COL]
    report["duplicate_feature_rows_kept"] = int(d.duplicated(subset=feat_cols).sum())

    d["tenure"] = pd.to_numeric(d["tenure"], errors="coerce")
    d["MonthlyCharges"] = pd.to_numeric(d["MonthlyCharges"], errors="coerce")
    tc = pd.to_numeric(d["TotalCharges"], errors="coerce")
    report["total_charges_blank"] = int(tc.isna().sum())
    est = (d["MonthlyCharges"] * d["tenure"]).where(d["tenure"] > 0, 0.0)
    d["TotalCharges"] = tc.fillna(est)           # tenure 0 -> 0, else monthly*tenure

    d["SeniorCitizen"] = pd.to_numeric(d["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    other_missing = d[RAW_FEATURES].isna().sum()
    report["other_missing_values"] = {k: int(v) for k, v in other_missing.items() if v}
    for c in CATEGORICAL_RAW:                     # impute rare gaps with the mode
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].mode().iloc[0])
    for c in ("tenure", "MonthlyCharges"):
        d[c] = d[c].fillna(d[c].median())

    y = d[settings.TARGET].astype("string").str.strip().str.lower()
    d[settings.TARGET] = y.map({"yes": 1, "no": 0, "1": 1, "0": 0})
    bad = int(d[settings.TARGET].isna().sum())
    report["rows_dropped_bad_target"] = bad
    d = d.dropna(subset=[settings.TARGET]).copy()
    d[settings.TARGET] = d[settings.TARGET].astype(int)

    report["rows_clean"] = int(len(d))
    return d.reset_index(drop=True), report


def outlier_report(df: pd.DataFrame, cols=NUMERIC_RAW) -> dict:
    """IQR (1.5x) outlier counts for numeric columns."""
    res = {}
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        res[c] = {"q1": float(q1), "q3": float(q3), "lower": float(lo),
                  "upper": float(hi), "n_outliers": n_out,
                  "pct": round(100 * n_out / len(df), 2)}
    return res


def class_balance(y) -> dict:
    y = pd.Series(y)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return {"n_churn": n_pos, "n_retained": n_neg,
            "churn_rate": round(n_pos / len(y), 4),
            "imbalance_ratio": round(n_neg / max(n_pos, 1), 3)}


# ------------------------------------------------------------ preprocessor
def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features, one-hot encode categoricals (unknown-safe)."""
    return ColumnTransformer([
        ("num", StandardScaler(), MODEL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         MODEL_CATEGORICAL),
    ], verbose_feature_names_out=True)


# -------------------------------------------------------- input validation
def validate_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise customer records used for prediction.

    * all required raw columns must exist
    * categorical values must be from the allowed sets
    * numeric values must be numeric and within sane ranges
    * a missing TotalCharges is estimated as MonthlyCharges * tenure
    Returns a cleaned copy; raises InputValidationError otherwise.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise InputValidationError(["No customer records supplied."])
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    problems = []

    required = [c for c in RAW_FEATURES if c != "TotalCharges"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise InputValidationError([f"Missing required column(s): {missing}"])

    for c in ["tenure", "MonthlyCharges"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "TotalCharges" not in d.columns:
        d["TotalCharges"] = np.nan
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d["SeniorCitizen"] = pd.to_numeric(d["SeniorCitizen"], errors="coerce")

    for c in ["tenure", "MonthlyCharges"]:
        n_bad = int(d[c].isna().sum())
        if n_bad:
            problems.append(f"'{c}' has {n_bad} missing/non-numeric value(s).")
    d["TotalCharges"] = d["TotalCharges"].fillna(d["MonthlyCharges"] * d["tenure"])

    for c, (lo, hi) in RANGES.items():
        bad = d[c].notna() & ((d[c] < lo) | (d[c] > hi))
        if bad.any():
            rows = (bad[bad].index[:5] + 1).tolist()
            problems.append(f"'{c}' must be between {lo} and {hi} (check row(s) {rows}).")

    for c in ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
              "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
              "PaperlessBilling", "PaymentMethod"]:
        d[c] = d[c].astype("string").str.strip()
        bad = ~d[c].isin(ALLOWED[c])
        if bad.any():
            vals = sorted(set(d.loc[bad, c].astype(str)))[:4]
            problems.append(f"'{c}' has invalid value(s) {vals}; allowed: {ALLOWED[c]}.")
    bad = ~d["SeniorCitizen"].isin([0, 1])
    if bad.any():
        problems.append("'SeniorCitizen' must be 0 or 1.")

    if problems:
        raise InputValidationError(problems)
    d["SeniorCitizen"] = d["SeniorCitizen"].astype(int)
    return d
