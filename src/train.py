"""End-to-end training: load -> clean -> EDA -> tune 4 models -> compare -> persist.

Usage:
    python -m src.train                       # real data if present, else synthetic file
    python -m src.train --generate-synthetic  # (re)create synthetic data first
    python -m src.train --data path/to.csv --n-iter 20
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from config import settings
from config.schema import FRIENDLY, RAW_FEATURES
from src import data_generation, data_loader, eda, evaluation
from src.feature_engineering import add_features
from src.preprocessing import (build_preprocessor, class_balance, clean_data,
                               outlier_report)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:                                   # pragma: no cover
    XGBClassifier, XGB_AVAILABLE = None, False


def make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("fe", FunctionTransformer(add_features, validate=False)),
        ("prep", build_preprocessor()),
        ("clf", clf),
    ])


def model_zoo(scale_pos_weight: float) -> dict:
    """name -> (estimator, param distribution). Class imbalance handled by weighting."""
    rs = settings.RANDOM_STATE
    zoo = {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=rs),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=rs),
            {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [10, 20, 40, 80],
             "clf__criterion": ["gini", "entropy"]}),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1,
                                   random_state=rs),
            {"clf__n_estimators": [150, 250], "clf__max_depth": [6, 8, 12],
             "clf__min_samples_leaf": [3, 8, 15], "clf__max_features": ["sqrt", 0.3]}),
    }
    if XGB_AVAILABLE:
        zoo["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=1,
                          scale_pos_weight=scale_pos_weight, random_state=rs),
            {"clf__n_estimators": [150, 300], "clf__max_depth": [3, 4, 5],
             "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.8, 1.0],
             "clf__colsample_bytree": [0.7, 1.0]})
    else:
        zoo["Gradient Boosting (XGBoost fallback)"] = (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=rs),
            {"clf__max_iter": [100, 200], "clf__max_depth": [3, 4, 6],
             "clf__learning_rate": [0.03, 0.05, 0.1], "clf__min_samples_leaf": [20, 50],
             "clf__l2_regularization": [0.0, 1.0]})
    return zoo


def run_training(data_path=None, n_iter=settings.N_SEARCH_ITER, generate_synthetic=False,
                 verbose=True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    for d in (settings.MODELS_DIR, settings.FIGURES_DIR, settings.PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if generate_synthetic and data_path is None and not settings.REAL_DATA_PATH.exists():
        data_generation.save_synthetic()
    raw, source = data_loader.load_raw_data(data_path)
    log(f"[1/6] Loaded {len(raw)} rows (source: {source.upper()})")
    if source == "synthetic":
        log("      WARNING: SYNTHETIC data - metrics are for pipeline testing only.")

    evaluation.set_figure_tag("SYNTHETIC DATA - for testing only" if source == "synthetic" else None)
    df, clean_report = clean_data(raw)
    df.to_csv(settings.CLEAN_DATA_PATH, index=False)
    balance = class_balance(df[settings.TARGET])
    outliers = outlier_report(df)
    eda.run_eda(df)
    log(f"[2/6] Cleaned -> {len(df)} rows; churn rate {balance['churn_rate']:.1%}")

    X, y = df[RAW_FEATURES], df[settings.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=settings.TEST_SIZE, stratify=y, random_state=settings.RANDOM_STATE)
    cv = StratifiedKFold(settings.CV_FOLDS, shuffle=True, random_state=settings.RANDOM_STATE)
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())

    results, fitted, probas, curves, best_params = [], {}, {}, {}, {}
    for name, (est, grid) in model_zoo(spw).items():
        log(f"[3/6] Tuning {name} ({n_iter} random configs x {settings.CV_FOLDS}-fold CV)...")
        n_combo = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(
            make_pipeline(est), grid, n_iter=min(n_iter, n_combo), cv=cv,
            scoring={"roc_auc": "roc_auc", "f1": "f1", "recall": "recall"},
            refit="f1", random_state=settings.RANDOM_STATE, n_jobs=1)
        search.fit(X_tr, y_tr)
        i = search.best_index_
        cvr = search.cv_results_
        proba = search.best_estimator_.predict_proba(X_te)[:, 1]
        m = evaluation.compute_metrics(y_te, proba, settings.DECISION_THRESHOLD)
        row = {"model": name,
               "cv_recall": float(cvr["mean_test_recall"][i]),
               "cv_f1": float(cvr["mean_test_f1"][i]),
               "cv_roc_auc": float(cvr["mean_test_roc_auc"][i]),
               "cv_f1_std": float(cvr["std_test_f1"][i]), **m}
        row["selection_score"] = (row["cv_recall"] + row["cv_f1"] + row["cv_roc_auc"]) / 3
        results.append(row)
        fitted[name], probas[name] = search.best_estimator_, proba
        best_params[name] = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                             for k, v in search.best_params_.items()}
        curves[name] = evaluation.curve_data(y_te, proba)
        log(f"      CV F1={row['cv_f1']:.3f}  test: recall={m['recall']:.3f} "
            f"F1={m['f1']:.3f} AUC={m['roc_auc']:.3f}")

    table = pd.DataFrame(results).sort_values("selection_score", ascending=False)
    best_name = table.iloc[0]["model"]
    best = fitted[best_name]
    log(f"[4/6] Best model (CV mean of recall, F1, ROC-AUC): {best_name}")

    # ---- figures & tables
    fig_dir = settings.FIGURES_DIR
    metrics_by_model = {r["model"]: r for r in results}
    evaluation.plot_roc(curves, metrics_by_model, fig_dir / "roc_curves.png")
    evaluation.plot_pr(curves, metrics_by_model, fig_dir / "pr_curves.png",
                       base_rate=float(y_te.mean()))
    evaluation.plot_model_comparison(table, fig_dir / "model_comparison.png")
    bm = metrics_by_model[best_name]
    cm = [[bm["tn"], bm["fp"]], [bm["fn"], bm["tp"]]]
    evaluation.plot_confusion_matrix(cm, fig_dir / "confusion_matrix.png",
                                     f"Confusion matrix - {best_name}")
    table.to_csv(settings.REPORTS_DIR / "model_comparison.csv", index=False)
    settings.CURVES_PATH.write_text(json.dumps(curves))

    log("[5/6] Computing permutation importance and explanations...")
    perm = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                  random_state=settings.RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"feature": [FRIENDLY.get(f, f) for f in RAW_FEATURES],
                        "importance": perm.importances_mean,
                        "std": perm.importances_std}
                       ).sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(settings.REPORTS_DIR / "feature_importance.csv", index=False)
    evaluation.plot_importance(imp, fig_dir / "feature_importance.png",
                               "Permutation importance (drop in ROC-AUC)")

    from src.explainability import ChurnExplainer
    bg = X_tr.sample(min(150, len(X_tr)), random_state=settings.RANDOM_STATE)
    bg.to_csv(settings.BACKGROUND_PATH, index=False)
    expl = ChurnExplainer(best, bg)
    g = expl.global_importance(X_te)
    g.to_csv(settings.REPORTS_DIR / "global_explanation.csv", index=False)
    evaluation.plot_importance(
        g, fig_dir / "global_explanation.png",
        f"Mean |contribution| per feature ({expl.backend_label}; {expl.unit})")

    # ---- persist
    joblib.dump(best, settings.MODEL_PATH)
    pred = (probas[best_name] >= settings.DECISION_THRESHOLD).astype(int)
    meta = {
        "project": "ChurnGuard AI",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source,
        "data_is_synthetic": source == "synthetic",
        "data_note": ("SYNTHETIC data for software testing only - not real customers."
                      if source == "synthetic" else "Real/custom dataset."),
        "xgboost_used": XGB_AVAILABLE,
        "explainer_backend": expl.backend_label, "explainer_unit": expl.unit,
        "best_model": best_name, "decision_threshold": settings.DECISION_THRESHOLD,
        "risk_thresholds": settings.RISK_THRESHOLDS,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "cv_folds": settings.CV_FOLDS, "search_iterations": n_iter,
        "cleaning": clean_report, "class_balance": balance, "outliers": outliers,
        "imbalance_handling": "class weights (balanced) / scale_pos_weight; no resampling",
        "models": results, "best_params": best_params,
        "test_metrics": bm, "confusion_matrix": cm,
        "classification_report": classification_report(
            y_te, pred, target_names=["Retained", "Churn"]),
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
        "train_seconds": round(time.time() - t0, 1),
    }
    settings.METADATA_PATH.write_text(json.dumps(meta, indent=2))
    log(f"[6/6] Saved model + metadata in {meta['train_seconds']}s -> {settings.MODEL_PATH}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=lambda p: __import__("pathlib").Path(p), default=None)
    ap.add_argument("--n-iter", type=int, default=settings.N_SEARCH_ITER)
    ap.add_argument("--generate-synthetic", action="store_true")
    a = ap.parse_args()
    try:
        run_training(a.data, a.n_iter, a.generate_synthetic)
    except data_loader.DatasetNotFoundError as e:
        raise SystemExit(str(e))
