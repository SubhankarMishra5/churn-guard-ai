"""Model loading and prediction helpers (single customer + batch)."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from config import settings
from config.schema import RAW_FEATURES
from src.preprocessing import validate_input_frame

SETUP_MODEL_HELP = (
    "Trained model not found at models/best_model.joblib.\n"
    "Train it first:\n"
    "    python -m src.train                      # uses data/raw/... if present\n"
    "    python -m src.train --generate-synthetic # synthetic data, testing only"
)


class ModelNotFoundError(FileNotFoundError):
    pass


def load_model(path=None):
    p = path or settings.MODEL_PATH
    if not p.exists():
        raise ModelNotFoundError(SETUP_MODEL_HELP)
    return joblib.load(p)


def load_metadata(path=None) -> dict:
    p = path or settings.METADATA_PATH
    if not p.exists():
        raise ModelNotFoundError(SETUP_MODEL_HELP)
    return json.loads(p.read_text())


def risk_level(prob: float, thresholds=None) -> str:
    t = thresholds or settings.RISK_THRESHOLDS
    return "Low" if prob < t["low"] else ("Medium" if prob < t["high"] else "High")


def predict_dataframe(df: pd.DataFrame, model=None, threshold=None) -> pd.DataFrame:
    """Validate, score and annotate a DataFrame of customers."""
    model = model or load_model()
    thr = settings.DECISION_THRESHOLD if threshold is None else threshold
    clean = validate_input_frame(df)
    proba = model.predict_proba(clean[RAW_FEATURES])[:, 1]
    out = df.copy().reset_index(drop=True)
    out["ChurnProbability"] = proba.round(4)
    out["ChurnPrediction"] = (proba >= thr).astype(int)
    out["RiskLevel"] = [risk_level(p) for p in proba]
    return out


def predict_single(record: dict, model=None, threshold=None) -> dict:
    res = predict_dataframe(pd.DataFrame([record]), model, threshold).iloc[0]
    return {"probability": float(res["ChurnProbability"]),
            "prediction": "Churn" if res["ChurnPrediction"] == 1 else "Retain",
            "risk": res["RiskLevel"]}


def retention_actions(record: dict, risk: str) -> list[str]:
    """Rule-based retention playbook tied to the customer's own attributes."""
    a = []
    if record.get("Contract") == "Month-to-month":
        a.append("Offer a discounted 1- or 2-year plan to move off month-to-month billing.")
    if record.get("PaymentMethod") == "Electronic check":
        a.append("Offer a small bill credit for switching to automatic card/bank payment.")
    if (record.get("InternetService") != "No"
            and record.get("TechSupport") != "Yes" and record.get("OnlineSecurity") != "Yes"):
        a.append("Bundle Tech Support and Online Security free for 3 months.")
    if float(record.get("tenure", 99)) <= 12:
        a.append("Trigger an onboarding check-in call and a first-year loyalty reward.")
    if float(record.get("MonthlyCharges", 0)) >= 80:
        a.append("Run a plan-optimisation review or price-lock to reduce bill shock.")
    if int(record.get("SeniorCitizen", 0)) == 1:
        a.append("Offer a dedicated support line and simplified billing.")
    if risk == "High":
        a.insert(0, "Priority outreach: assign a retention specialist within 48 hours.")
    if not a:
        a.append("Maintain engagement: send satisfaction survey and tailored upsell offers.")
    return a[:5]
