"""Generate docs/MODEL_EVALUATION.md from the real training artifacts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from config import settings


def main():
    meta = json.loads(settings.METADATA_PATH.read_text())
    imp = pd.read_csv(settings.REPORTS_DIR / "feature_importance.csv").head(8)
    tm, syn = meta["test_metrics"], meta["data_is_synthetic"]
    L = ["# Model Evaluation", ""]
    if syn:
        L += ["> **SYNTHETIC DATA WARNING.** These numbers were produced on a simulated dataset "
              "(same schema as IBM Telco) for software testing. They are **not** real-world results. "
              "Run `python -m src.train` with the real CSV and `python scripts/build_all.py` to "
              "regenerate this file.", ""]
    L += [f"- Trained: {meta['trained_at']}", f"- Data source: **{meta['data_source']}**",
          f"- Rows: {meta['cleaning']['rows_clean']:,} (train {meta['n_train']:,} / test {meta['n_test']:,}, stratified)",
          f"- Churn rate: {meta['class_balance']['churn_rate']:.1%}; imbalance handled by: {meta['imbalance_handling']}",
          f"- Tuning: randomised search, {meta['search_iterations']} configs, stratified {meta['cv_folds']}-fold CV, refit on F1",
          f"- Selection rule: highest mean of CV recall, F1, ROC-AUC -> **{meta['best_model']}**",
          f"- XGBoost installed during training: {meta['xgboost_used']}"
          + ("" if meta["xgboost_used"] else " (scikit-learn HistGradientBoosting used as fallback)"),
          f"- Explainer backend: {meta['explainer_backend']} ({meta['explainer_unit']})", "",
          "## Model comparison", "",
          "| Model | CV recall | CV F1 | CV ROC-AUC | Test recall | Test F1 | Test ROC-AUC | Test precision | Test accuracy |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(meta["models"], key=lambda r: -r["selection_score"]):
        L.append(f"| {m['model']} | {m['cv_recall']:.3f} | {m['cv_f1']:.3f} | {m['cv_roc_auc']:.3f} | "
                 f"{m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['precision']:.3f} | {m['accuracy']:.3f} |")
    L += ["", f"## Selected model: {meta['best_model']}", "",
          f"Confusion matrix (test): TN={tm['tn']}, FP={tm['fp']}, FN={tm['fn']}, TP={tm['tp']}", "",
          "```", meta["classification_report"].rstrip(), "```", "",
          "Best hyper-parameters:", "", "```json", json.dumps(meta["best_params"], indent=2), "```", "",
          "## Top features (permutation importance, drop in ROC-AUC)", "",
          "| Feature | Importance |", "|---|---|"]
    L += [f"| {r.feature} | {r.importance:.4f} |" for r in imp.itertuples()]
    L += ["", "Figures: `reports/figures/` (confusion matrix, ROC, precision-recall, comparison, "
          "importance, EDA). Interpretation limits: differences between top models are small "
          "relative to fold-to-fold variation; probabilities are not calibrated; the 0.50 "
          "threshold is a default, not cost-optimised."]
    out = ROOT / "docs" / "MODEL_EVALUATION.md"
    out.write_text("\n".join(L) + "\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
