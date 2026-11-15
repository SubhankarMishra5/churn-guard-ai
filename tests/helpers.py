"""Shared test fixtures: a tiny synthetic dataset and a quickly-trained pipeline."""
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config.schema import RAW_FEATURES
from src.data_generation import generate_synthetic_telco
from src.preprocessing import clean_data
from src.train import make_pipeline

_cache = {}


def tiny_data(n=900):
    if "data" not in _cache:
        raw = generate_synthetic_telco(n=n, seed=7, n_duplicates=5)
        _cache["data"] = (raw, *clean_data(raw))
    return _cache["data"]


def tiny_model():
    if "model" not in _cache:
        _, clean, _ = tiny_data()
        m = make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(clean[RAW_FEATURES], clean["Churn"])
        _cache["model"] = m
    return _cache["model"]


def sample_record(**over):
    rec = dict(gender="Female", SeniorCitizen=0, Partner="Yes", Dependents="No", tenure=12,
               PhoneService="Yes", MultipleLines="No", InternetService="DSL",
               OnlineSecurity="No", OnlineBackup="Yes", DeviceProtection="No",
               TechSupport="No", StreamingTV="No", StreamingMovies="No",
               Contract="Month-to-month", PaperlessBilling="Yes",
               PaymentMethod="Mailed check", MonthlyCharges=50.0)
    rec.update(over)
    return rec
