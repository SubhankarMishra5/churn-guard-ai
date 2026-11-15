"""ChurnGuard AI - Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import settings
from config.schema import ALLOWED, FRIENDLY, RAW_FEATURES
from src import analytics, predict
from src.explainability import ChurnExplainer, explanation_text
from src.preprocessing import InputValidationError

NAVY, ORANGE, GREEN, AMBER = "#1F4E79", "#E4572E", "#2E933C", "#F2A541"
RISK_COLORS = {"Low": GREEN, "Medium": AMBER, "High": ORANGE}
PAGES = ["Dashboard", "Customer Prediction", "Analytics", "Model Performance",
         "Batch Prediction"]

CSS = """
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {background: rgba(31,78,121,0.07); border-radius: 10px;
    padding: 12px 16px; border-left: 5px solid #1F4E79;}
.cg-hero {background: linear-gradient(90deg,#1F4E79,#2d6fa8); color: white;
    padding: 18px 24px; border-radius: 12px; margin-bottom: 14px;}
.cg-hero h1 {color: white; margin: 0; font-size: 1.9rem;}
.cg-hero p {margin: 4px 0 0 0; opacity: 0.9;}
</style>
"""


# ------------------------------------------------------------------ loaders
@st.cache_resource(show_spinner=False)
def get_model():
    return predict.load_model()


@st.cache_data(show_spinner=False)
def get_meta() -> dict:
    return predict.load_metadata()


@st.cache_data(show_spinner=False)
def get_clean() -> pd.DataFrame:
    return pd.read_csv(settings.CLEAN_DATA_PATH)


@st.cache_data(show_spinner=False)
def get_scored() -> pd.DataFrame:
    df = get_clean()
    scored = predict.predict_dataframe(df.drop(columns=[settings.TARGET]), get_model())
    scored[settings.TARGET] = df[settings.TARGET].values
    return scored


@st.cache_resource(show_spinner=False)
def get_explainer():
    return ChurnExplainer(get_model(), pd.read_csv(settings.BACKGROUND_PATH))


def artifacts_ready() -> bool:
    return all(p.exists() for p in (settings.MODEL_PATH, settings.METADATA_PATH,
                                    settings.CLEAN_DATA_PATH, settings.BACKGROUND_PATH))


# ------------------------------------------------------------------- setup
def page_setup():
    st.error("Trained model or processed data not found - one-time setup required.")
    st.markdown("**Option A - real data (recommended)**")
    st.code(f"# 1. download the IBM Telco Customer Churn CSV\n"
            f"# 2. save it as: {settings.REAL_DATA_PATH}\n"
            "python -m src.train", language="bash")
    st.markdown("**Option B - synthetic data (software testing only)**")
    st.code("python -m src.train --generate-synthetic", language="bash")
    if st.button("Generate SYNTHETIC data and train now (quick, testing only)"):
        from src import train
        with st.spinner("Training models (a few minutes)..."):
            train.run_training(generate_synthetic=True, n_iter=4, verbose=False)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# --------------------------------------------------------------- dashboard
def page_dashboard():
    meta, scored = get_meta(), get_scored()
    k = analytics.kpis(scored)
    tm = meta["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{k['total']:,}")
    c2.metric("Predicted churn customers", f"{k['predicted_churn']:,}")
    c3.metric("Predicted churn %", f"{k['predicted_churn_pct']}%")
    c4.metric("High-risk customers", f"{k['high_risk']:,}")
    st.caption("Predictions are made on the full customer file. Because the model saw "
               "part of this data during training, use the hold-out metrics below "
               "(not these counts) to judge accuracy.")

    st.subheader(f"Model performance - {meta['best_model']} (hold-out test set)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recall", f"{tm['recall']:.3f}")
    m2.metric("F1-score", f"{tm['f1']:.3f}")
    m3.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    m4.metric("Precision", f"{tm['precision']:.3f}")
    m5.metric("Accuracy", f"{tm['accuracy']:.3f}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(scored, x="ChurnProbability", nbins=30, color="RiskLevel",
                           color_discrete_map=RISK_COLORS,
                           title="Distribution of churn probability")
        st.plotly_chart(fig)
    with right:
        order = ["Low", "Medium", "High"]
        rc = scored["RiskLevel"].value_counts().reindex(order).fillna(0).reset_index()
        rc.columns = ["RiskLevel", "customers"]
        st.plotly_chart(px.bar(rc, x="RiskLevel", y="customers", color="RiskLevel",
                               color_discrete_map=RISK_COLORS, title="Customers by risk level"))

    st.subheader("Top 15 customers to contact first")
    cols = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges",
                        "ChurnProbability", "RiskLevel"] if c in scored.columns]
    st.dataframe(scored.sort_values("ChurnProbability", ascending=False)[cols].head(15))


# -------------------------------------------------------------- prediction
def _customer_form() -> dict | None:
    with st.form("customer_form"):
        st.markdown("**Account**")
        a1, a2, a3 = st.columns(3)
        tenure = a1.number_input("Tenure (months)", min_value=0, max_value=120, value=6)
        monthly = a2.number_input("Monthly charges", min_value=0.0, max_value=500.0,
                                  value=79.85, step=0.5)
        total = a3.number_input("Total charges (0 = estimate)", min_value=0.0,
                                max_value=100000.0, value=0.0, step=10.0)
        b1, b2, b3 = st.columns(3)
        contract = b1.selectbox("Contract", ALLOWED["Contract"])
        payment = b2.selectbox("Payment method", ALLOWED["PaymentMethod"])
        paperless = b3.selectbox("Paperless billing", ALLOWED["PaperlessBilling"])
        st.markdown("**Services**")
        s1, s2, s3 = st.columns(3)
        internet = s1.selectbox("Internet service", ALLOWED["InternetService"], index=1)
        phone = s2.selectbox("Phone service", ALLOWED["PhoneService"])
        lines = s3.selectbox("Multiple lines", ALLOWED["MultipleLines"], index=1)
        addons = {}
        cols = st.columns(3)
        for i, name in enumerate(["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                  "TechSupport", "StreamingTV", "StreamingMovies"]):
            addons[name] = cols[i % 3].selectbox(FRIENDLY[name], ALLOWED[name], index=1)
        st.markdown("**Demographics**")
        d1, d2, d3, d4 = st.columns(4)
        gender = d1.selectbox("Gender", ALLOWED["gender"])
        senior = d2.selectbox("Senior citizen (1 = yes)", ALLOWED["SeniorCitizen"])
        partner = d3.selectbox("Partner", ALLOWED["Partner"], index=1)
        dependents = d4.selectbox("Dependents", ALLOWED["Dependents"], index=1)
        submitted = st.form_submit_button("Predict churn risk")
    if not submitted:
        return None
    rec = {"gender": gender, "SeniorCitizen": senior, "Partner": partner,
           "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
           "MultipleLines": lines, "InternetService": internet, **addons,
           "Contract": contract, "PaperlessBilling": paperless,
           "PaymentMethod": payment, "MonthlyCharges": monthly}
    if total > 0:
        rec["TotalCharges"] = total
    return rec


def page_prediction():
    st.write("Enter a customer's details to estimate churn probability.")
    rec = _customer_form()
    if rec is None:
        return
    model, meta = get_model(), get_meta()
    try:
        res = predict.predict_single(rec, model, meta["decision_threshold"])
    except InputValidationError as e:
        st.error("Please fix the following: " + "; ".join(e.problems))
        return
    prob, risk = res["probability"], res["risk"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", res["prediction"])
    c2.metric("Churn probability", f"{prob:.1%}")
    c3.metric("Risk level", risk)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[risk]},
               "steps": [{"range": [0, settings.RISK_THRESHOLDS["low"] * 100], "color": "#dff2e1"},
                         {"range": [settings.RISK_THRESHOLDS["low"] * 100,
                                    settings.RISK_THRESHOLDS["high"] * 100], "color": "#fdebc8"},
                         {"range": [settings.RISK_THRESHOLDS["high"] * 100, 100],
                          "color": "#f9d3c8"}]}))
    st.plotly_chart(gauge)

    st.subheader("Why this prediction?")
    clean = predict.validate_input_frame(pd.DataFrame([rec]))
    expl = get_explainer()
    contrib = expl.local(clean)
    for line in explanation_text(contrib):
        st.write("- " + line)
    top = contrib.head(10).copy()
    top["contribution"] = top["contribution"].astype(float)
    fig = px.bar(top.iloc[::-1], x="contribution", y="label", orientation="h",
                 color="direction", title=f"Top drivers ({expl.backend_label}; {expl.unit})",
                 color_discrete_map={"raises risk": ORANGE, "lowers risk": GREEN})
    st.plotly_chart(fig)

    st.subheader("Suggested retention actions")
    for action in predict.retention_actions({**rec}, risk):
        st.write("- " + action)


# --------------------------------------------------------------- analytics
def page_analytics():
    df = get_clean()
    f1, f2 = st.columns(2)
    contracts = f1.multiselect("Filter: contract", ALLOWED["Contract"],
                               default=ALLOWED["Contract"])
    nets = f2.multiselect("Filter: internet service", ALLOWED["InternetService"],
                          default=ALLOWED["InternetService"])
    d = df[df["Contract"].isin(contracts) & df["InternetService"].isin(nets)]
    if d.empty:
        st.warning("No customers match the selected filters.")
        return
    st.caption(f"{len(d):,} customers selected - churn rate "
               f"{100 * d[settings.TARGET].mean():.1f}%")

    def bar(tbl, x, title):
        fig = px.bar(tbl, x=x, y="churn_rate", text="churn_rate", hover_data=["customers"],
                     title=title, color_discrete_sequence=[NAVY])
        fig.update_layout(yaxis_title="Churn rate (%)")
        return fig

    t1, t2, t3, t4, t5 = st.tabs(["Tenure", "Contract", "Payment method",
                                  "Monthly charges", "Segments"])
    with t1:
        st.plotly_chart(bar(analytics.tenure_group(d), "TenureGroup",
                            "Churn rate by tenure group (months)"))
    with t2:
        st.plotly_chart(bar(analytics.churn_rate_by(d, "Contract"), "Contract",
                            "Churn rate by contract type"))
    with t3:
        st.plotly_chart(bar(analytics.churn_rate_by(d, "PaymentMethod"), "PaymentMethod",
                            "Churn rate by payment method"))
    with t4:
        st.plotly_chart(bar(analytics.charge_band(d), "ChargeBand",
                            "Churn rate by monthly-charge band"))
        dd = d.assign(Status=d[settings.TARGET].map({1: "Churn", 0: "Retained"}))
        st.plotly_chart(px.histogram(dd, x="MonthlyCharges", color="Status", nbins=40,
                                     barmode="overlay", opacity=0.65,
                                     title="Monthly charges distribution"))
    with t5:
        seg = analytics.segment_table(d)
        st.plotly_chart(bar(seg, "Segment", "Churn rate by customer segment"))
        st.dataframe(seg)


# ------------------------------------------------------ model performance
def page_performance():
    meta = get_meta()
    models = pd.DataFrame(meta["models"])
    if meta.get("data_is_synthetic"):
        st.info("These metrics come from SYNTHETIC data and only demonstrate that the "
                "pipeline works. Train on the real dataset for genuine results.")
    st.caption(f"Best model: **{meta['best_model']}** - selected by mean cross-validated "
               "recall, F1 and ROC-AUC. Imbalance handled with class weights.")
    t1, t2, t3, t4, t5 = st.tabs(["Comparison", "Confusion matrix", "ROC & PR curves",
                                  "Feature importance", "Explainability"])
    with t1:
        show = models[["model", "recall", "f1", "roc_auc", "precision", "accuracy",
                       "cv_f1", "selection_score"]].round(4)
        st.dataframe(show)
        long = show.melt(id_vars="model", value_vars=["recall", "f1", "roc_auc",
                                                       "precision", "accuracy"])
        st.plotly_chart(px.bar(long, x="model", y="value", color="variable", barmode="group",
                               title="Hold-out test metrics"))
    with t2:
        cm = meta["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Retained", "Churn"], y=["Retained", "Churn"],
                        color_continuous_scale="Blues",
                        labels={"x": "Predicted", "y": "Actual"},
                        title=f"Confusion matrix - {meta['best_model']}")
        st.plotly_chart(fig)
        st.text(meta["classification_report"])
    with t3:
        curves = json.loads(settings.CURVES_PATH.read_text())
        roc, pr = go.Figure(), go.Figure()
        for name, c in curves.items():
            roc.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=name))
            pr.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=name))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                 line={"dash": "dash", "color": "gray"}))
        roc.update_layout(title="ROC curves", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        pr.update_layout(title="Precision-recall curves", xaxis_title="Recall",
                         yaxis_title="Precision")
        st.plotly_chart(roc)
        st.plotly_chart(pr)
    with t4:
        imp = pd.read_csv(settings.REPORTS_DIR / "feature_importance.csv").head(15)
        st.plotly_chart(px.bar(imp.iloc[::-1], x="importance", y="feature", orientation="h",
                               title="Permutation importance (drop in ROC-AUC)"))
    with t5:
        st.write(f"Explanation method: **{meta['explainer_backend']}** "
                 f"(unit: {meta['explainer_unit']}).")
        g = pd.read_csv(settings.REPORTS_DIR / "global_explanation.csv").head(15)
        st.plotly_chart(px.bar(g.iloc[::-1], x="importance", y="feature", orientation="h",
                               title="Global feature influence (mean |contribution|)"))
        st.caption("Per-customer explanations appear on the Customer Prediction page.")


# ------------------------------------------------------------------- batch
def page_batch():
    st.write("Upload a CSV with one customer per row. Required columns: "
             + ", ".join(c for c in RAW_FEATURES if c != "TotalCharges")
             + " (TotalCharges optional; estimated when missing).")
    template = get_clean().drop(columns=[settings.TARGET]).head(5)
    st.download_button("Download CSV template", template.to_csv(index=False).encode(),
                       file_name="churnguard_template.csv", mime="text/csv")
    up = st.file_uploader("Customer CSV", type=["csv"])
    if up is None:
        return
    try:
        df = pd.read_csv(up)
    except Exception as e:                       # unreadable / not a CSV
        st.error(f"Could not read the file as CSV: {e}")
        return
    try:
        out = predict.predict_dataframe(df, get_model(), get_meta()["decision_threshold"])
    except InputValidationError as e:
        st.error("The file failed validation:")
        for p in e.problems:
            st.write("- " + p)
        return
    k = analytics.kpis(out)
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", f"{k['total']:,}")
    c2.metric("Predicted to churn", f"{k['predicted_churn']:,}")
    c3.metric("High risk", f"{k['high_risk']:,}")
    st.dataframe(out.sort_values("ChurnProbability", ascending=False).head(200))
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                       file_name="churnguard_predictions.csv", mime="text/csv")


# -------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="cg-hero"><h1>ChurnGuard AI</h1>'
                '<p>Intelligent customer churn prediction and retention intelligence</p>'
                '</div>', unsafe_allow_html=True)
    if not artifacts_ready():
        page_setup()
        return
    meta = get_meta()
    if meta.get("data_is_synthetic"):
        st.warning("Running on SYNTHETIC test data - numbers are illustrative only, "
                   "not real customer results.")
    page = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.caption(f"Model: {meta['best_model']}  \nTrained: {meta['trained_at'][:10]}  \n"
                       f"Data: {meta['data_source']}")
    {"Dashboard": page_dashboard, "Customer Prediction": page_prediction,
     "Analytics": page_analytics, "Model Performance": page_performance,
     "Batch Prediction": page_batch}[page]()


if __name__ == "__main__":
    main()
