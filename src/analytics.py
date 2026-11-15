"""Dashboard analytics helpers (pure pandas - unit-testable, UI-agnostic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings


def churn_rate_by(df: pd.DataFrame, col: str, target: str = settings.TARGET) -> pd.DataFrame:
    g = df.groupby(col, observed=True)[target].agg(["count", "sum", "mean"]).reset_index()
    g.columns = [col, "customers", "churned", "churn_rate"]
    g["churn_rate"] = (g["churn_rate"] * 100).round(2)
    return g


def tenure_group(df):
    d = df.copy()
    d["TenureGroup"] = pd.cut(d["tenure"], [-1, 12, 24, 48, 72, np.inf],
                              labels=["0-12", "13-24", "25-48", "49-72", "72+"])
    return churn_rate_by(d, "TenureGroup")


def charge_band(df):
    d = df.copy()
    d["ChargeBand"] = pd.cut(d["MonthlyCharges"], [0, 35, 60, 85, 200],
                             labels=["<35", "35-60", "60-85", ">85"])
    return churn_rate_by(d, "ChargeBand")


def assign_segment(row) -> str:
    if row["tenure"] <= 12 and row["Contract"] == "Month-to-month":
        return "New month-to-month"
    if row["InternetService"] == "Fiber optic" and row["MonthlyCharges"] >= 80:
        return "High-value fiber"
    if row["Contract"] != "Month-to-month" and row["tenure"] > 24:
        return "Loyal long-contract"
    if row["SeniorCitizen"] == 1:
        return "Senior"
    return "Standard"


def segment_table(df):
    d = df.copy()
    d["Segment"] = d.apply(assign_segment, axis=1)
    return churn_rate_by(d, "Segment").sort_values("churn_rate", ascending=False)


def kpis(scored: pd.DataFrame) -> dict:
    n = len(scored)
    pred = int(scored["ChurnPrediction"].sum())
    return {"total": n, "predicted_churn": pred,
            "predicted_churn_pct": round(100 * pred / max(n, 1), 2),
            "high_risk": int((scored["RiskLevel"] == "High").sum())}
