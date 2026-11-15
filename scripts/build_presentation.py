"""Build the 10-slide presentation (+ speaker notes + speaking script) from real results.

    python scripts/build_presentation.py [--author "Your Name"]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from config import settings

NAVY, ORANGE = RGBColor(0x1F, 0x4E, 0x79), RGBColor(0xE4, 0x57, 0x2E)
LIGHT, DARK, WHITE = RGBColor(0xEE, 0xF2, 0xF7), RGBColor(0x22, 0x2B, 0x36), RGBColor(255, 255, 255)
RED = RGBColor(0xB0, 0x00, 0x20)
FIG = settings.FIGURES_DIR
FONT = "Calibri"


def tb(slide, x, y, w, h, text="", size=16, bold=False, color=DARK, align=PP_ALIGN.LEFT,
       anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.name = Pt(size), bold, FONT
    r.font.color.rgb = color
    return box


def bullets(slide, x, y, w, h, items, size=18, gap=8):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = "\u2022  " + it
        r.font.size, r.font.name = Pt(size), FONT
        r.font.color.rgb = DARK
    return box


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def header(slide, title, n, syn):
    rect(slide, 0, 0, 13.333, 1.05, NAVY)
    tb(slide, 0.5, 0.12, 12.2, 0.8, title, 30, True, WHITE, anchor=MSO_ANCHOR.MIDDLE)
    tb(slide, 12.3, 7.05, 0.8, 0.3, str(n), 11, color=RGBColor(0x66, 0x66, 0x66), align=PP_ALIGN.RIGHT)
    if syn:
        tb(slide, 0.5, 7.05, 9, 0.3, "Results on SYNTHETIC data - pipeline demonstration only",
           11, True, RED)


def stat(slide, x, y, w, label, value, fill=NAVY):
    rect(slide, x, y, w, 1.25, fill)
    tb(slide, x, y + 0.1, w, 0.65, value, 28, True, WHITE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    tb(slide, x, y + 0.77, w, 0.4, label, 13, False, WHITE, PP_ALIGN.CENTER)


def picture(slide, path, x, y, max_w, max_h):
    from PIL import Image
    w, h = Image.open(path).size
    scale = min(max_w / w, max_h / h)
    pw, ph = w * scale, h * scale
    slide.shapes.add_picture(str(path), Inches(x + (max_w - pw) / 2), Inches(y + (max_h - ph) / 2),
                             Inches(pw), Inches(ph))


def f3(x):
    return f"{x:.3f}"


def build(author, out_pptx, out_script):
    meta = json.loads(settings.METADATA_PATH.read_text())
    syn, tm, cb, cl = meta["data_is_synthetic"], meta["test_metrics"], meta["class_balance"], meta["cleaning"]
    best = meta["best_model"]
    models = sorted(meta["models"], key=lambda r: -r["selection_score"])
    notes = []

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    def new(title, n, note):
        s = prs.slides.add_slide(blank)
        header(s, title, n, syn)
        s.notes_slide.notes_text_frame.text = note
        notes.append((title, note))
        return s

    # 1 ---------------------------------------------------------------- title
    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    rect(s, 0.8, 3.55, 1.6, 0.08, ORANGE)
    tb(s, 0.8, 1.6, 11.5, 1.1, "ChurnGuard AI", 54, True, WHITE)
    tb(s, 0.8, 2.7, 11.5, 0.8, "Intelligent Customer Churn Prediction and Retention Intelligence System",
       24, False, WHITE)
    tb(s, 0.8, 3.9, 11, 0.5, f"Internship Project  |  {author}", 18, False, WHITE)
    tb(s, 0.8, 4.4, 11, 0.5, "Python  |  scikit-learn  |  SHAP-ready explainability  |  Streamlit", 16,
       False, RGBColor(0xC9, 0xD8, 0xEA))
    if syn:
        rect(s, 0.8, 5.6, 11.7, 0.9, RGBColor(0xFD, 0xEC, 0xEA))
        tb(s, 0.95, 5.65, 11.4, 0.8,
           "All figures in this deck come from a SYNTHETIC dataset (same schema as IBM Telco) "
           "because the real file could not be downloaded in the build environment. "
           "They demonstrate the pipeline, not real-world performance.", 14, True, RED,
           anchor=MSO_ANCHOR.MIDDLE)
    t1 = ("Good morning. I'm presenting ChurnGuard AI, an end-to-end system that predicts which "
          "telecom customers are likely to leave, explains why, and suggests how to keep them. "
          + ("One honesty point up front: the numbers you'll see today were produced on a synthetic "
             "dataset that mirrors the IBM Telco schema, because I had no internet access to download "
             "the real file during the build. They prove the pipeline works; they are not claims about "
             "real customers. Re-running one command on the real CSV regenerates everything." if syn else
             "All results use the IBM Telco Customer Churn dataset."))
    s.notes_slide.notes_text_frame.text = t1
    notes.append(("Title", t1))

    # 2 ------------------------------------------------------------- problem
    s = new("The problem: customers leave, and it is costly", 2,
            "Churn means a customer cancels their service. Winning a new customer usually costs more "
            "than keeping an existing one, so companies want to find at-risk customers early. The "
            "challenge is that churners are a minority - about one in four here - so a model that "
            "simply says nobody will leave looks 74 percent accurate and is useless. That is why we "
            "focus on recall and F1 alongside ROC-AUC, not accuracy alone.")
    bullets(s, 0.6, 1.4, 7.4, 5.2, [
        "Acquiring a customer typically costs more than retaining one",
        f"Churners are a minority: {cb['churn_rate']:.1%} of customers",
        "Accuracy alone is misleading on imbalanced data",
        "Goal: predict churn probability, explain it, and recommend an action",
        "Priority metrics: recall, F1-score and ROC-AUC"], 24, 20)
    stat(s, 8.6, 1.6, 4.1, "of customers churn", f"{cb['churn_rate']:.1%}")
    stat(s, 8.6, 3.2, 4.1, "retained : churn ratio", f"{cb['imbalance_ratio']:.1f} : 1", ORANGE)
    stat(s, 8.6, 4.8, 4.1, "customers analysed", f"{cl['rows_clean']:,}")

    # 3 ------------------------------------------------------------ approach
    s = new("Objectives and technology stack", 3,
            "The project has five goals: a reproducible preprocessing pipeline, four compared models "
            "with cross-validation and tuning, proper evaluation, explanations for each prediction, "
            "and a usable dashboard backed by automated tests. The stack is standard Python data "
            "science tooling with Streamlit for the interface.")
    tb(s, 0.6, 1.3, 6, 0.5, "Objectives", 24, True, NAVY)
    bullets(s, 0.6, 1.9, 6.2, 4.8, [
        "Clean, validated, leakage-safe pipeline",
        "Compare 4 models with stratified 5-fold CV and tuning",
        "Evaluate: accuracy, precision, recall, F1, ROC-AUC, PR",
        "Explain every prediction",
        "Dashboard for single and batch scoring",
        "Automated tests"], 22, 16)
    tb(s, 7.2, 1.3, 5.5, 0.5, "Stack", 24, True, NAVY)
    bullets(s, 7.2, 1.9, 5.6, 4.8, [
        "Python, pandas, NumPy",
        "scikit-learn Pipelines",
        "Logistic Regression, Decision Tree, Random Forest, "
        + ("XGBoost" if meta["xgboost_used"] else "gradient boosting (XGBoost fallback)"),
        "SHAP (with built-in fallback attribution)",
        "Streamlit + Plotly dashboard",
        "joblib persistence, unittest"], 22, 16)

    # 4 ---------------------------------------------------------------- data
    s = new("Dataset and preprocessing", 4,
            ("The intended dataset is IBM Telco Customer Churn with about seven thousand customers. "
             "In this run I used a synthetic dataset with the same columns, deliberately including "
             "duplicates and blank total-charge values so the cleaning code is exercised. " if syn else
             "The dataset is the IBM Telco Customer Churn data. ") +
            f"Cleaning removed {cl['duplicates_removed']} duplicates and repaired "
            f"{cl['total_charges_blank']} blank TotalCharges values, leaving {cl['rows_clean']:,} "
            "customers. I engineered features such as add-on service count, month-to-month flag, "
            "auto-pay flag and tenure group. Scaling and one-hot encoding live inside the sklearn "
            "pipeline so no information leaks from the test set. Imbalance is handled with class "
            "weights rather than resampling.")
    stat(s, 0.6, 1.4, 2.9, "raw rows", f"{cl['rows_raw']:,}")
    stat(s, 3.7, 1.4, 2.9, "duplicates removed", f"{cl['duplicates_removed']}", ORANGE)
    stat(s, 6.8, 1.4, 2.9, "blank charges repaired", f"{cl['total_charges_blank']}")
    stat(s, 9.9, 1.4, 2.9, "clean rows", f"{cl['rows_clean']:,}", ORANGE)
    bullets(s, 0.6, 3.1, 12.2, 3.8, [
        "Validation: schema check, whitespace trimming, duplicate removal, missing-value repair",
        "Outliers: 1.5 x IQR analysis; legitimate large totals retained",
        "Features: add-on count, month-to-month, auto-pay, no-protection, charge ratio, tenure group",
        "Scaling + one-hot encoding inside the pipeline (no leakage); stratified 80/20 split",
        "Imbalance: balanced class weights / scale_pos_weight"], 22, 16)

    # 5 --------------------------------------------------------- architecture
    s = new("System architecture", 5,
            "On the top is the training pipeline: data, validation and cleaning, feature engineering, "
            "preprocessing, model training with cross-validation, then evaluation, selection and "
            "persistence of the best model plus metadata. Below is the serving layer: prediction, "
            "explainability and analytics modules feeding the Streamlit app. Because the whole "
            "pipeline is saved as one object, prediction applies exactly the same steps as training.")
    picture(s, settings.ASSETS_DIR / "architecture.png", 0.6, 1.25, 12.1, 5.7)

    # 6 ---------------------------------------------------------------- models
    s = new("Models and methodology", 6,
            "Four models were tuned with randomised search and stratified five-fold cross-validation, "
            "refit on F1. I selected the winner by the average of cross-validated recall, F1 and "
            "ROC-AUC rather than test performance, so the test set stays an honest final check. "
            + ("The gradient boosting entry is scikit-learn's HistGradientBoosting because XGBoost "
               "could not be installed offline; the code uses real XGBoost automatically when it is "
               "available. " if not meta["xgboost_used"] else "") +
            f"{best} ranked first, but the models are close - differences are within fold-to-fold noise.")
    rows = [["Model", "CV recall", "CV F1", "CV AUC", "Test recall", "Test F1", "Test AUC"]]
    for m in models:
        rows.append([m["model"], f3(m["cv_recall"]), f3(m["cv_f1"]), f3(m["cv_roc_auc"]),
                     f3(m["recall"]), f3(m["f1"]), f3(m["roc_auc"])])
    shape = s.shapes.add_table(len(rows), 7, Inches(0.6), Inches(1.4), Inches(12.1), Inches(2.6))
    tbl = shape.table
    tbl.columns[0].width = Inches(4.0)
    for c in range(1, 7):
        tbl.columns[c].width = Inches(1.35)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = val
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            para.runs[0].font.size, para.runs[0].font.name = Pt(15), FONT
            para.runs[0].font.bold = (r == 0 or (r == 1))
            para.runs[0].font.color.rgb = WHITE if r == 0 else DARK
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (LIGHT if r % 2 else WHITE)
    bullets(s, 0.6, 4.4, 12.1, 2.5, [
        f"Selected: {best} (highest mean of CV recall, F1, ROC-AUC); row 1 above",
        f"{meta['search_iterations']}-config randomised search, stratified {meta['cv_folds']}-fold CV, refit on F1",
        "Selection uses CV only; the test set is held out for the final report"], 18, 8)

    # 7 --------------------------------------------------------------- results
    s = new("Results: model comparison", 7,
            f"On the held-out test set, {best} reached recall {f3(tm['recall'])}, F1 {f3(tm['f1'])} and "
            f"ROC-AUC {f3(tm['roc_auc'])}. The bar chart shows all four models are close. "
            + ("Because the data is synthetic and its churn rule is logistic, a logistic model being "
               "best is expected, and this ranking should not be generalised to real data." if syn else
               "Differences between the top models are small relative to sampling noise."))
    picture(s, FIG / "model_comparison.png", 0.4, 1.3, 8.2, 5.4)
    stat(s, 9.0, 1.4, 3.8, "Recall", f3(tm["recall"]))
    stat(s, 9.0, 2.9, 3.8, "F1-score", f3(tm["f1"]), ORANGE)
    stat(s, 9.0, 4.4, 3.8, "ROC-AUC", f3(tm["roc_auc"]))
    tb(s, 9.0, 5.8, 3.8, 0.8, f"Selected: {best}", 15, True, NAVY, PP_ALIGN.CENTER)

    # 8 ------------------------------------------------------------ evaluation
    s = new("Evaluation of the selected model", 8,
            f"The confusion matrix shows {tm['tp']} churners caught and {tm['fn']} missed, with "
            f"{tm['fp']} false alarms. Class weighting deliberately favours recall, so precision is "
            f"about {tm['precision']:.0%}. Whether that trade is acceptable depends on the cost of a "
            "retention offer versus the value of a saved customer, and the threshold can be adjusted. "
            "The ROC curves show all models clearly above chance.")
    picture(s, FIG / "confusion_matrix.png", 0.5, 1.3, 5.0, 4.4)
    picture(s, FIG / "roc_curves.png", 5.7, 1.3, 7.1, 5.0)
    tb(s, 0.5, 5.8, 5.2, 1.0,
       f"Precision {f3(tm['precision'])}  |  Recall {f3(tm['recall'])}  |  Accuracy {f3(tm['accuracy'])}",
       15, True, NAVY, PP_ALIGN.CENTER)

    # 9 ------------------------------------------------- explainability + app
    s = new("Explainable AI and the dashboard", 9,
            f"Each prediction is explained by per-feature contributions. This run used the "
            f"{meta['explainer_backend']} method; the SHAP code path is implemented but was not "
            "executed here because SHAP could not be installed offline. The chart shows the global "
            "drivers. The Streamlit dashboard has five pages: Dashboard, Customer Prediction with "
            "risk level, reasons and retention actions, interactive Analytics, Model Performance, "
            "and Batch Prediction with CSV upload and download.")
    picture(s, FIG / "global_explanation.png", 0.4, 1.3, 6.6, 5.4)
    tb(s, 7.3, 1.3, 5.5, 0.5, "Dashboard pages", 22, True, NAVY)
    bullets(s, 7.3, 1.9, 5.6, 4.8, [
        "Dashboard: KPIs, risk distribution, top at-risk customers",
        "Prediction: probability, Low/Medium/High risk, top drivers, retention actions",
        "Analytics: churn by tenure, contract, payment, charges, segment",
        "Model Performance: comparison, confusion matrix, ROC/PR, importance",
        "Batch: upload CSV, validate, download predictions"], 17, 9)

    # 10 ------------------------------------------------------------ conclusion
    s = new("Conclusion, limitations and future work", 10,
            "To summarise: a complete, tested churn workflow with four compared models, evaluation "
            "centred on recall, F1 and ROC-AUC, explanations and a dashboard. The main limitation "
            + ("is that results are on synthetic data, so the first next step is to train on the real "
               "IBM Telco file and regenerate the report. " if syn else "is a single static dataset. ") +
            "Other future work: cost-sensitive thresholds, probability calibration, uplift modelling, "
            "fairness auditing and deployment behind an API. Thank you - happy to take questions.")
    tb(s, 0.6, 1.3, 6, 0.5, "Achieved", 22, True, NAVY)
    bullets(s, 0.6, 1.9, 6.0, 4.9, [
        "End-to-end pipeline, 4 models, CV + tuning",
        "Recall / F1 / ROC-AUC-driven selection",
        "Explanations + retention recommendations",
        "5-page dashboard, single and batch scoring",
        "Automated tests (unit + page smoke tests)"], 21, 14)
    tb(s, 7.0, 1.3, 6, 0.5, "Limitations and next steps", 22, True, NAVY)
    bullets(s, 7.0, 1.9, 5.8, 4.9, ([
        "Train on the real IBM Telco data (current results are synthetic)"] if syn else []) + [
        "Cost-sensitive threshold; probability calibration",
        "Time-based validation; drift monitoring",
        "Fairness audit; API + Docker deployment"], 21, 14)

    prs.save(out_pptx)
    lines = ["# ChurnGuard AI - Slide-by-Slide Speaking Script", "",
             "Approximate total time: 8-10 minutes (about 45-60 seconds per slide).", ""]
    for i, (title, note) in enumerate(notes, 1):
        lines += [f"## Slide {i}: {title}", "", note, ""]
    Path(out_script).write_text("\n".join(lines))
    return out_pptx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="[Your Name]")
    ap.add_argument("--out", default=str(settings.REPORTS_DIR / "ChurnGuard_AI_Presentation.pptx"))
    ap.add_argument("--script", default=str(ROOT / "docs" / "Presentation_Speaking_Script.md"))
    a = ap.parse_args()
    print("Wrote", build(a.author, a.out, a.script))
