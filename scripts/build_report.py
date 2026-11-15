"""Build the academic project report (PDF) from the real training artifacts.

    python scripts/build_report.py [--author "Your Name"] [--institution "..."]

Every number comes from models/metadata.json and reports/*.csv, so re-running after
training on the real dataset regenerates the report with real results.
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from config import settings

NAVY = colors.HexColor("#1F4E79")
LIGHT = colors.HexColor("#EEF2F7")
FIG = settings.FIGURES_DIR

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10.2,
                      leading=14.6, alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=16,
                    leading=20, textColor=NAVY, spaceBefore=6, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                    leading=16, textColor=NAVY, spaceBefore=8, spaceAfter=4, keepWithNext=1)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.6, leading=11, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444444"), spaceAfter=10)
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=8.4, leading=11,
                      alignment=0, backColor=LIGHT, borderPadding=4, spaceAfter=8)
WARN = ParagraphStyle("warn", parent=BODY, backColor=colors.HexColor("#FDECEA"),
                      borderColor=colors.HexColor("#B00020"), borderWidth=1, borderPadding=7,
                      textColor=colors.HexColor("#7A0016"), spaceBefore=6, spaceAfter=10)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2,
                     alignment=0)

fig_no = [0]
tab_no = [0]


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="\u2022") for i in items] + [Spacer(1, 4)]


def figure(path, caption, width_cm=13.5):
    path = Path(path)
    if not path.exists():
        return []
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    fig_no[0] += 1
    wd = width_cm * cm
    return [KeepTogether([Image(str(path), width=wd, height=wd * h / w),
                          P(f"Figure {fig_no[0]}. {caption}", CAP)])]


def table(rows, caption, col_widths=None, font=8.6):
    tab_no[0] += 1
    cell = ParagraphStyle("cell", parent=BODY, fontSize=font, leading=font + 2.4,
                          alignment=0, spaceAfter=0)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    def wrap(r, ri):
        return [Paragraph(str(c).replace("&", "&amp;") if "<" not in str(c) else str(c),
                          head if ri == 0 else cell) for c in r]
    t = Table([wrap(r, i) for i, r in enumerate(rows)], colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B8C4D6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return [KeepTogether([P(f"Table {tab_no[0]}. {caption}", CAP), t, Spacer(1, 10)])]


def f3(x):
    return f"{x:.3f}"


def run_tests():
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                       cwd=ROOT, capture_output=True, text=True)
    out = r.stderr + r.stdout
    m = re.search(r"Ran (\d+) tests? in ([\d.]+)s", out)
    ok = "OK" in out.splitlines()[-1] if out.strip() else False
    return (int(m.group(1)) if m else 0), (m.group(2) if m else "?"), ok


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.2 * cm, "ChurnGuard AI - Project Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build(author, institution, out_path):
    meta = json.loads(settings.METADATA_PATH.read_text())
    models = pd.DataFrame(meta["models"]).sort_values("selection_score", ascending=False)
    imp = pd.read_csv(settings.REPORTS_DIR / "feature_importance.csv")
    gexp = pd.read_csv(settings.REPORTS_DIR / "global_explanation.csv")
    syn = meta["data_is_synthetic"]
    tm, cb, cl = meta["test_metrics"], meta["class_balance"], meta["cleaning"]
    best = meta["best_model"]
    n_tests, t_secs, tests_ok = run_tests()
    xgb_note = ("" if meta["xgboost_used"] else
                " The XGBoost package was not installable in the build environment, so "
                "scikit-learn's HistGradientBoostingClassifier (also a gradient-boosted tree "
                "ensemble) was trained in its place and is labelled accordingly throughout. "
                "Re-running training where xgboost is installed trains true XGBoost "
                "automatically.")
    S = []

    # ------------------------------------------------------------------ cover
    S += [Spacer(1, 3.2 * cm),
          P("ChurnGuard AI", ParagraphStyle("t", parent=H1, fontSize=34, leading=40, alignment=TA_CENTER)),
          P("Intelligent Customer Churn Prediction and Retention Intelligence System",
            ParagraphStyle("st", parent=H2, fontSize=15, leading=20, alignment=TA_CENTER)),
          Spacer(1, 0.8 * cm), P("Internship Project Report", ParagraphStyle(
              "x", parent=BODY, fontSize=13, alignment=TA_CENTER)),
          Spacer(1, 2.2 * cm),
          P(f"<b>Submitted by:</b> {author}", ParagraphStyle("c", parent=BODY, alignment=TA_CENTER)),
          P(f"<b>Institution:</b> {institution}", ParagraphStyle("c", parent=BODY, alignment=TA_CENTER)),
          P(f"<b>Date:</b> {date.today():%d %B %Y}", ParagraphStyle("c", parent=BODY, alignment=TA_CENTER))]
    if syn:
        S += [Spacer(1, 1.2 * cm), P(
            "<b>IMPORTANT - READ BEFORE USING ANY NUMBER IN THIS REPORT.</b> The quantitative "
            "results were produced on a <b>synthetic, simulated</b> dataset with the same schema "
            "as the IBM Telco Customer Churn data, because the build environment had no internet "
            "access to download the real dataset. They demonstrate that the implementation works "
            "end to end; they are <b>not</b> evidence about real customers. Re-run "
            "<font face='Courier'>python -m src.train</font> with the real CSV in "
            "<font face='Courier'>data/raw/</font>, then "
            "<font face='Courier'>python scripts/build_report.py</font>, to regenerate this report "
            "with genuine results.", WARN)]
    S.append(PageBreak())

    # --------------------------------------------------------------- contents
    S += [P("Contents", H1)]
    sections = ["Abstract", "Introduction", "Problem Statement", "Objectives", "Literature Review",
                "Dataset Description", "System Architecture", "Methodology", "Algorithms Used",
                "Implementation", "Results and Discussion", "Model Evaluation", "Explainable AI",
                "Conclusion", "Future Scope", "References"]
    S += [P(f"{i}. {s}", ParagraphStyle("toc", parent=BODY, alignment=0, spaceAfter=3))
          for i, s in enumerate(sections, 1)] + [PageBreak()]

    # --------------------------------------------------------------- 1 Abstract
    S += [P("1. Abstract", H1), P(
        "Customer churn - customers discontinuing a subscription service - is costly because "
        "acquiring a new customer is generally more expensive than retaining an existing one. "
        "This project presents ChurnGuard AI, an end-to-end machine learning system that predicts "
        "the probability that a telecom customer will churn, explains the prediction, and "
        "recommends retention actions. The system covers data validation and cleaning, feature "
        "engineering, class-imbalance handling, comparison of four classifiers (Logistic "
        "Regression, Decision Tree, Random Forest and a gradient-boosted tree model) using "
        "stratified 5-fold cross-validation and randomised hyper-parameter search, and a "
        "five-page Streamlit dashboard for single and batch prediction, analytics, model "
        "performance and explainability."
        f" Because identifying churners matters more than raw accuracy, models were selected on the "
        f"mean of cross-validated recall, F1-score and ROC-AUC. On the hold-out test set the "
        f"selected model, <b>{best}</b>, achieved recall {f3(tm['recall'])}, F1 {f3(tm['f1'])} "
        f"and ROC-AUC {f3(tm['roc_auc'])}"
        + (" on <b>synthetic</b> data, which validates the pipeline but says nothing about "
           "real-world performance." if syn else ".")
        + f" The code base is modular and covered by {n_tests} automated tests.")]

    # ------------------------------------------------------------ 2 Introduction
    S += [P("2. Introduction", H1), P(
        "Subscription businesses such as telecommunications, streaming and software-as-a-service "
        "depend on recurring revenue. When customers leave, the business loses not only future "
        "revenue but also the investment made in acquiring them. Retention programmes - discounts, "
        "service upgrades, proactive outreach - are effective only if they are directed at the "
        "customers who are actually at risk. Machine learning allows a company to score every "
        "customer with a churn probability so that limited retention budgets can be focused on "
        "the highest-risk accounts."),
        P("This report documents the design, implementation and evaluation of ChurnGuard AI, "
          "built with Python, scikit-learn and Streamlit, with explainability and a retention "
          "recommendation layer so that results are usable by non-technical business staff.")]

    # ------------------------------------------------------------ 3 Problem
    S += [P("3. Problem Statement", H1), P(
        "Given a customer's demographics, subscribed services, contract and billing details, "
        "tenure and charges, predict whether the customer will churn (binary classification) and "
        "estimate the churn probability. The solution must (a) cope with class imbalance, because "
        "churners are a minority; (b) prioritise recall and F1-score alongside ROC-AUC, because "
        "missing a churner is costly; (c) explain each prediction; and (d) be usable by "
        "non-programmers through a web interface supporting both single and batch scoring.")]

    # ------------------------------------------------------------ 4 Objectives
    S += [P("4. Objectives", H1)] + bullets([
        "Build a reproducible pipeline for data validation, cleaning, feature engineering and "
        "preprocessing.",
        "Train and compare at least four models with stratified cross-validation and "
        "hyper-parameter tuning, and persist the best one.",
        "Evaluate with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC and "
        "precision-recall curves.",
        "Provide per-customer and global explanations of predictions.",
        "Deliver an interactive dashboard (dashboard, prediction, analytics, model performance, "
        "batch prediction) backed by automated tests."])

    # ------------------------------------------------------------ 5 Literature
    S += [P("5. Literature Review", H1), P(
        "<b>Business motivation.</b> Reichheld and Sasser [1] argued that small improvements in "
        "customer retention can produce disproportionately large profit gains, which motivates "
        "targeted retention spending. Verbeke et al. [2] reviewed churn prediction in "
        "telecommunications and showed that model evaluation should reflect business value rather "
        "than only statistical accuracy - a perspective that supports this project's emphasis on "
        "recall and on actionable risk bands."),
        P("<b>Models.</b> Logistic regression [3] is a transparent baseline whose coefficients are "
          "interpretable. Decision trees are easy to visualise but unstable; Random Forests [4] "
          "reduce variance by averaging many de-correlated trees and are strong on tabular data. "
          "Gradient-boosted trees, popularised by XGBoost [5], build trees sequentially to correct "
          "earlier errors and frequently top tabular benchmarks."),
        P("<b>Evaluation under imbalance.</b> Accuracy can be misleading when one class "
          "dominates. ROC analysis [6] summarises the true/false-positive trade-off across "
          "thresholds, while precision-recall curves [7] are more informative when the positive "
          "class is rare. Resampling such as SMOTE [8] is one remedy; this project instead uses "
          "class weighting, which avoids creating synthetic training records and keeps the "
          "cross-validation folds clean."),
        P("<b>Explainability.</b> SHAP [9] attributes a prediction to its input features using "
          "Shapley values from cooperative game theory, giving locally accurate and consistent "
          "explanations. This project integrates SHAP and also provides a model-agnostic "
          "fallback attribution for environments where SHAP is not installed.")]

    # ------------------------------------------------------------ 6 Dataset
    S += [P("6. Dataset Description", H1)]
    if syn:
        S += [P("<b>Dataset actually used in this edition: synthetic.</b> The intended dataset is "
                "the publicly available IBM Telco Customer Churn dataset [10] (7,043 customers, "
                "21 columns, roughly 26-27% churn). The build environment had no internet access, "
                "so the real file could not be downloaded and <b>no claim is made that it was "
                "used</b>. Instead, <font face='Courier'>src/data_generation.py</font> simulated "
                "7,043 customers (plus 15 injected duplicate rows and 11 blank TotalCharges "
                "values, to exercise the cleaning code) using the same column names and "
                "categories. Churn was drawn from a documented logistic rule (month-to-month "
                "contracts, fibre internet, electronic-check payment, short tenure and high "
                "charges raise risk). Consequently the models are, by construction, recovering a "
                "known rule, and the Logistic Regression family has a built-in advantage.", WARN)]
    else:
        S += [P("The IBM Telco Customer Churn dataset [10] was used.")]
    S += [P("Each record describes one customer: demographics (gender, senior citizen, partner, "
            "dependents), account information (tenure, contract, payment method, paperless "
            "billing, monthly and total charges), subscribed services (phone, multiple lines, "
            "internet type, online security, backup, device protection, tech support, streaming "
            "TV and movies) and the target <i>Churn</i> (Yes/No). The customer identifier is "
            "dropped before modelling.")]
    S += table([["Item", "Value"],
                ["Raw rows", f"{cl['rows_raw']:,}"],
                ["Duplicate rows removed", f"{cl['duplicates_removed']:,}"],
                ["Blank TotalCharges repaired", f"{cl['total_charges_blank']:,}"],
                ["Clean rows used", f"{cl['rows_clean']:,}"],
                ["Churners / retained", f"{cb['n_churn']:,} / {cb['n_retained']:,}"],
                ["Churn rate", f"{cb['churn_rate']:.1%}"],
                ["Imbalance ratio (retained:churn)", f"{cb['imbalance_ratio']:.2f} : 1"],
                ["Train / test rows (stratified 80/20)", f"{meta['n_train']:,} / {meta['n_test']:,}"]],
               "Dataset summary after cleaning.", [8 * cm, 6 * cm])
    out_rows = [["Feature", "Lower fence", "Upper fence", "Outliers", "% of rows"]]
    for k, v in meta["outliers"].items():
        out_rows.append([k, f"{v['lower']:.1f}", f"{v['upper']:.1f}", v["n_outliers"], f"{v['pct']}%"])
    S += [P("Outliers were assessed with the 1.5 x IQR rule. Flagged values were retained because "
            "large totals are legitimate for long-tenure customers and the models used "
            "(scaled linear and tree-based) are not unduly sensitive to them.")]
    S += table(out_rows, "IQR outlier analysis of numeric features.",
               [4 * cm, 2.8 * cm, 2.8 * cm, 2.4 * cm, 2.4 * cm])
    S += figure(FIG / "eda_class_distribution.png", "Class distribution.", 8)
    S += figure(FIG / "eda_tenure_charges.png", "Tenure distribution and monthly charges by churn.", 14)
    S += figure(FIG / "eda_churn_by_category.png", "Churn rate by contract type and payment method.", 15)
    S += figure(FIG / "eda_correlation.png", "Correlation of numeric features with churn.", 9)

    # ------------------------------------------------------------ 7 Architecture
    S += [P("7. System Architecture", H1), P(
        "The system has a training pipeline and a serving layer. The training pipeline validates "
        "and cleans data, engineers features, builds a scikit-learn Pipeline (feature engineering "
        "-> scaling and one-hot encoding -> classifier), tunes four models, selects the best and "
        "persists it with metadata. The serving layer loads the saved pipeline, validates user "
        "input, produces probabilities and risk bands, explains them and presents everything in "
        "a Streamlit dashboard. Because the whole pipeline is serialised, prediction applies "
        "exactly the same transformations as training.")]
    S += figure(settings.ASSETS_DIR / "architecture.png", "ChurnGuard AI architecture.", 16)

    # ------------------------------------------------------------ 8 Methodology
    S += [P("8. Methodology", H1)] + bullets([
        "<b>Validation and cleaning:</b> schema check; whitespace trimming; exact-duplicate "
        "removal; blank TotalCharges repaired (0 when tenure is 0, otherwise monthly charge x "
        "tenure); target mapped to 0/1.",
        "<b>Feature engineering:</b> number of add-on services, internet flag, month-to-month "
        "flag, auto-pay flag, no-protection flag (internet without security or tech support), "
        "average monthly spend, charge ratio, and tenure group.",
        "<b>Preprocessing:</b> StandardScaler for numeric features; one-hot encoding with "
        "unknown-category handling for categorical features. Both sit inside the pipeline, so "
        "scaling statistics are learned only from training folds (no leakage).",
        f"<b>Imbalance:</b> churners are {cb['churn_rate']:.1%} of customers. Class weights "
        "('balanced') or scale_pos_weight were used rather than resampling.",
        f"<b>Splitting and tuning:</b> stratified 80/20 train-test split; "
        f"{meta['search_iterations']}-configuration randomised search per model with stratified "
        f"{meta['cv_folds']}-fold cross-validation, refit on F1.",
        "<b>Selection:</b> the model with the highest mean of cross-validated recall, F1 and "
        "ROC-AUC (not test-set performance, to avoid selecting on the test data).",
        "<b>Decision threshold:</b> 0.50. Churn probability is mapped to Low (&lt;0.30), Medium "
        "(0.30-0.60) and High (&ge;0.60) risk."])

    # ------------------------------------------------------------ 9 Algorithms
    S += [P("9. Algorithms Used", H1)]
    S += table([["Model", "Role", "Tuned parameters (best found)"]] + [
        [m, role, ", ".join(f"{k}={v}" for k, v in meta["best_params"][m].items())]
        for m, role in zip(meta["best_params"],
                           ["Interpretable linear baseline", "Single, visualisable tree",
                            "Bagged ensemble of trees", "Boosted tree ensemble"])],
        "Models compared and best hyper-parameters.", [3.6 * cm, 3.6 * cm, 9.4 * cm], 7.8)
    S += [P("Logistic regression models the log-odds of churn as a linear function of the "
            "features. A decision tree recursively splits on the most informative feature. A "
            "random forest averages many trees trained on bootstrap samples with random feature "
            "subsets. Gradient boosting adds shallow trees one at a time, each fitted to the "
            "errors of the ensemble so far." + xgb_note)]

    # ------------------------------------------------------------ 10 Implementation
    S += [P("10. Implementation", H1), P(
        "The project is a modular Python package. Key modules are listed below.")]
    S += table([["Module", "Responsibility"],
                ["config/", "Paths, hyper-parameters, risk bands, expected schema"],
                ["src/data_loader.py, data_generation.py", "Dataset discovery, schema validation, synthetic test data"],
                ["src/preprocessing.py", "Cleaning, outlier report, preprocessor, input validation"],
                ["src/feature_engineering.py", "Stateless feature creation"],
                ["src/train.py", "Training, tuning, comparison, persistence, figures"],
                ["src/evaluation.py", "Metrics, confusion matrix, ROC/PR plots"],
                ["src/predict.py", "Model loading, single/batch prediction, risk bands, retention rules"],
                ["src/explainability.py", "SHAP and fallback attribution; plain-English reasons"],
                ["src/analytics.py", "Churn by tenure/contract/payment/charges/segment"],
                ["app.py", "Streamlit dashboard (5 pages)"],
                ["tests/", "Automated unit and smoke tests"]],
               "Code organisation.", [5.6 * cm, 11 * cm])
    S += [P("<b>Dashboard.</b> Dashboard (KPIs, risk distribution, top at-risk customers); "
            "Customer Prediction (probability, risk, drivers, retention actions); Analytics "
            "(interactive churn charts with filters); Model Performance (comparison, confusion "
            "matrix, ROC/PR, feature importance, explanations); Batch Prediction (CSV upload, "
            "validation messages, CSV download)."),
          P("<b>Testing.</b> " + (
              f"{n_tests} automated tests ran in {t_secs}s and "
              f"{'all passed' if tests_ok else 'did NOT all pass - see test output'} at report "
              "build time. They cover cleaning, feature engineering, input validation, model "
              "loading, single and batch prediction, explanations and analytics. A smoke test "
              "executes every dashboard page against a stubbed Streamlit; it verifies the "
              "Python logic but is not a browser rendering test.")),
          P("<b>Robustness.</b> If the dataset or model is missing, scripts and the app print "
            "or display explicit setup instructions instead of failing obscurely.")]

    # ------------------------------------------------------------ 11 Results
    S += [P("11. Results and Discussion", H1)]
    rows = [["Model", "CV recall", "CV F1", "CV AUC", "Test recall", "Test F1", "Test AUC"]]
    for _, r in models.iterrows():
        rows.append([r["model"], f3(r["cv_recall"]), f3(r["cv_f1"]), f3(r["cv_roc_auc"]),
                     f3(r["recall"]), f3(r["f1"]), f3(r["roc_auc"])])
    S += table(rows, "Cross-validation (5-fold, training data) and hold-out test results.",
               [5.4 * cm, 1.8 * cm, 1.6 * cm, 1.7 * cm, 2.0 * cm, 1.7 * cm, 1.8 * cm], 8)
    spread = models["roc_auc"].max() - models["roc_auc"].min()
    S += [P(
        f"{best} ranked first on the selection score. The hold-out ROC-AUC values of the four "
        f"models span only {spread:.3f}, and cross-validated F1 varies by roughly "
        f"+/-{models.loc[models['model'] == best, 'cv_f1_std'].iloc[0]:.3f} between folds, so "
        "differences between the top models are small relative to sampling noise and should not "
        "be over-interpreted."
        + (" On this synthetic data the churn rule is itself logistic, which naturally favours "
           "Logistic Regression; on real data tree ensembles often match or exceed it, so the "
           "ranking here must not be generalised." if syn else ""))]
    S += figure(FIG / "model_comparison.png", "Hold-out metrics for all models.", 14)

    # ------------------------------------------------------------ 12 Evaluation
    S += [P("12. Model Evaluation", H1), P(
        f"Evaluation of <b>{best}</b> on {meta['n_test']:,} held-out customers at threshold "
        f"{meta['decision_threshold']}:")]
    S += table([["Metric", "Value"], ["Accuracy", f3(tm["accuracy"])],
                ["Precision", f3(tm["precision"])], ["Recall", f3(tm["recall"])],
                ["F1-score", f3(tm["f1"])], ["ROC-AUC", f3(tm["roc_auc"])],
                ["PR-AUC (average precision)", f3(tm["pr_auc"])]],
               "Hold-out test metrics of the selected model.", [7 * cm, 4 * cm])
    S += [P(f"The confusion matrix shows {tm['tp']} true positives, {tm['fn']} missed churners, "
            f"{tm['fp']} false alarms and {tm['tn']} true negatives. Class weighting trades "
            "precision for recall: the model flags more customers than will actually leave, so "
            f"only about {tm['precision']:.0%} of flagged customers are true churners. Whether "
            "that is acceptable depends on the cost of a retention offer versus the value of a "
            "retained customer; the decision threshold can be tuned accordingly.")]
    S += [Preformatted(meta["classification_report"], ParagraphStyle(
        "pre", parent=CODE, leading=10.5, fontSize=8)), Spacer(1, 6)]
    S += figure(FIG / "confusion_matrix.png", "Confusion matrix of the selected model.", 8)
    S += figure(FIG / "roc_curves.png", "ROC curves of all models.", 11)
    S += figure(FIG / "pr_curves.png", "Precision-recall curves of all models.", 11)

    # ------------------------------------------------------------ 13 XAI
    S += [P("13. Explainable AI", H1), P(
        f"Explanations in this run used the <b>{meta['explainer_backend']}</b> backend "
        f"(unit: {meta['explainer_unit']}). "
        + ("SHAP was not installed in the build environment, so the model-agnostic fallback was "
           "used: for each feature, the customer's value is replaced by values from a background "
           "sample and the average change in churn probability is measured. This is a "
           "single-feature attribution; unlike SHAP it ignores feature interactions. The SHAP "
           "code path is implemented (TreeExplainer / LinearExplainer, with one-hot columns "
           "summed back to each original feature) but was <b>not executed</b> here; it activates "
           "automatically when the shap package is installed."
           if meta["explainer_backend"] != "SHAP" else
           "SHAP TreeExplainer / LinearExplainer values were aggregated per original feature."))]
    top_perm = ", ".join(imp["feature"].head(4))
    top_glob = ", ".join(gexp["feature"].head(4))
    S += [P(f"Permutation importance (drop in ROC-AUC on the test set) ranks {top_perm} highest; "
            f"mean absolute local contribution ranks {top_glob} highest. Two independent methods "
            "agreeing on the main drivers increases confidence that the model relies on the "
            "intended signals."
            + (" This agrees with the simulated rule used to generate the data, which is a useful "
               "correctness check of the pipeline." if syn else ""))]
    S += figure(FIG / "feature_importance.png", "Permutation feature importance.", 11)
    S += figure(FIG / "global_explanation.png", "Global feature influence from per-customer attributions.", 11)
    S += [P("For an individual customer the dashboard lists the top drivers in plain English "
            "(e.g. 'Contract type = Month-to-month increases this customer's churn risk') and "
            "shows a bar chart of contributions, followed by rule-based retention actions.")]

    # ------------------------------------------------------------ 14 Conclusion
    S += [P("14. Conclusion", H1), P(
        "ChurnGuard AI delivers a complete, tested churn-prediction workflow: a leakage-safe "
        "preprocessing pipeline, four tuned and cross-validated models, evaluation focused on "
        "recall, F1 and ROC-AUC, per-customer explanations, retention recommendations and a "
        "five-page dashboard supporting single and batch scoring."
        + (" All reported performance figures were obtained on synthetic data and demonstrate "
           "pipeline correctness only; the real IBM Telco dataset should be used to obtain "
           "genuine performance figures, which the supplied scripts regenerate automatically."
           if syn else ""))]
    S += [P("<b>Limitations.</b>")] + bullets([
        "Single dataset, single snapshot; no temporal validation (customers observed over time).",
        "Predictions show association, not causation; retention actions are rule-based "
        "heuristics, not tested interventions.",
        "No cost-sensitive threshold optimisation; the 0.50 threshold is a default.",
        "Probabilities are not explicitly calibrated.",
        "The dashboard scores the training population, so its churn counts are optimistic; use "
        "the hold-out metrics to judge accuracy.",
        "Potential fairness issues (e.g. senior-citizen or gender features) were not audited."])

    # ------------------------------------------------------------ 15 Future
    S += [P("15. Future Scope", H1)] + bullets([
        "Train on the real IBM Telco data and benchmark against the results here.",
        "Cost-sensitive threshold selection using retention-offer cost and customer lifetime value.",
        "Probability calibration and uplift modelling to target customers who respond to offers.",
        "Time-based validation and drift monitoring in production.",
        "Fairness audit, REST API deployment (FastAPI), Docker packaging and CI/CD."])

    # ------------------------------------------------------------ 16 References
    S += [P("16. References", H1)]
    refs = [
        "F. F. Reichheld and W. E. Sasser, \"Zero defections: Quality comes to services,\" "
        "<i>Harvard Business Review</i>, vol. 68, no. 5, 1990.",
        "W. Verbeke, K. Dejaeger, D. Martens, J. Hur and B. Baesens, \"New insights into churn "
        "prediction in the telecommunication sector: A profit driven data mining approach,\" "
        "<i>European Journal of Operational Research</i>, vol. 218, no. 1, 2012.",
        "D. W. Hosmer, S. Lemeshow and R. X. Sturdivant, <i>Applied Logistic Regression</i>, "
        "3rd ed., Wiley, 2013.",
        "L. Breiman, \"Random forests,\" <i>Machine Learning</i>, vol. 45, no. 1, pp. 5-32, 2001.",
        "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in <i>Proc. 22nd "
        "ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining</i>, 2016, pp. 785-794.",
        "T. Fawcett, \"An introduction to ROC analysis,\" <i>Pattern Recognition Letters</i>, "
        "vol. 27, no. 8, pp. 861-874, 2006.",
        "T. Saito and M. Rehmsmeier, \"The precision-recall plot is more informative than the ROC "
        "plot when evaluating binary classifiers on imbalanced datasets,\" <i>PLoS ONE</i>, "
        "vol. 10, no. 3, e0118432, 2015.",
        "N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, \"SMOTE: Synthetic "
        "minority over-sampling technique,\" <i>Journal of Artificial Intelligence Research</i>, "
        "vol. 16, pp. 321-357, 2002.",
        "S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" "
        "in <i>Advances in Neural Information Processing Systems 30</i>, 2017, pp. 4765-4774.",
        "IBM Sample Data Sets, \"Telco Customer Churn\" (also mirrored on Kaggle).",
        "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" <i>Journal of Machine "
        "Learning Research</i>, vol. 12, pp. 2825-2830, 2011."]
    S += [Paragraph(r, BUL, bulletText=f"[{i}]") for i, r in enumerate(refs, 1)]

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="ChurnGuard AI - Internship Project Report", author=author)
    doc.build(S, onFirstPage=lambda c, d: None, onLaterPages=footer)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="[Your Name]")
    ap.add_argument("--institution", default="[Your Institution]")
    ap.add_argument("--out", default=str(settings.REPORTS_DIR / "ChurnGuard_AI_Project_Report.pdf"))
    a = ap.parse_args()
    print("Wrote", build(a.author, a.institution, Path(a.out)))
