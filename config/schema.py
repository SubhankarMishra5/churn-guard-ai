"""Expected raw schema (IBM Telco Customer Churn layout)."""

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
_ADDON_VALUES = ["Yes", "No", "No internet service"]

ALLOWED = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{c: list(_ADDON_VALUES) for c in ADDON_COLS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

RANGES = {"tenure": (0, 120), "MonthlyCharges": (0.0, 500.0),
          "TotalCharges": (0.0, 100000.0)}

NUMERIC_RAW = list(RANGES)
CATEGORICAL_RAW = [c for c in ALLOWED if c != "SeniorCitizen"]
RAW_FEATURES = list(ALLOWED) + NUMERIC_RAW

# Engineered columns (created in src/feature_engineering.py)
ENGINEERED_NUMERIC = ["NumAddOnServices", "AvgMonthlySpend", "ChargeRatio",
                      "HasInternet", "IsMonthToMonth", "AutoPay", "NoProtection"]
ENGINEERED_CATEGORICAL = ["TenureGroup"]

MODEL_NUMERIC = NUMERIC_RAW + ["SeniorCitizen"] + ENGINEERED_NUMERIC
MODEL_CATEGORICAL = CATEGORICAL_RAW + ENGINEERED_CATEGORICAL

FRIENDLY = {
    "tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges", "Contract": "Contract type",
    "PaymentMethod": "Payment method", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "PaperlessBilling": "Paperless billing", "SeniorCitizen": "Senior citizen",
    "MultipleLines": "Multiple lines", "PhoneService": "Phone service",
    "Partner": "Has partner", "Dependents": "Has dependents", "gender": "Gender",
    "NumAddOnServices": "Number of add-on services",
    "AvgMonthlySpend": "Average monthly spend", "ChargeRatio": "Charge ratio",
    "HasInternet": "Has internet", "IsMonthToMonth": "Month-to-month contract",
    "AutoPay": "Automatic payment", "NoProtection": "No security/support add-ons",
    "TenureGroup": "Tenure group",
}
