"""Stateless feature engineering (safe to run at training and prediction time)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import ADDON_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tenure = pd.to_numeric(out["tenure"], errors="coerce").fillna(0)
    monthly = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0)

    out["NumAddOnServices"] = sum((out[c] == "Yes").astype(int) for c in ADDON_COLS)
    out["HasInternet"] = (out["InternetService"] != "No").astype(int)
    out["IsMonthToMonth"] = (out["Contract"] == "Month-to-month").astype(int)
    out["AutoPay"] = out["PaymentMethod"].astype(str).str.contains("automatic").astype(int)
    out["NoProtection"] = ((out["InternetService"] != "No")
                           & (out["OnlineSecurity"] != "Yes")
                           & (out["TechSupport"] != "Yes")).astype(int)
    avg = total / tenure.clip(lower=1)
    out["AvgMonthlySpend"] = avg
    ratio = (monthly / avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["ChargeRatio"] = ratio.fillna(1.0)
    out["TenureGroup"] = pd.cut(tenure, bins=[-1, 12, 24, 48, 72, np.inf],
                                labels=["0-12", "13-24", "25-48", "49-72", "72+"]
                                ).astype(str)
    return out
