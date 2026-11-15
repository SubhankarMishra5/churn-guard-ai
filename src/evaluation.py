"""Metrics and plots for model evaluation."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

PALETTE = ["#1F4E79", "#E4572E", "#2E933C", "#F2A541", "#7A4EAB"]
FIG_TAG = None          # e.g. "SYNTHETIC DATA - for testing only"; stamped on every figure


def set_figure_tag(tag):
    global FIG_TAG
    FIG_TAG = tag


def stamp(fig):
    if FIG_TAG:
        fig.text(0.995, 0.005, FIG_TAG, ha="right", va="bottom", fontsize=7,
                 color="#B00020", style="italic")


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def curve_data(y_true, proba, max_points: int = 200) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    def thin(a, b):
        idx = np.linspace(0, len(a) - 1, min(max_points, len(a))).astype(int)
        return np.asarray(a)[idx].round(5).tolist(), np.asarray(b)[idx].round(5).tolist()
    f, t = thin(fpr, tpr)
    r, p = thin(rec, prec)
    return {"fpr": f, "tpr": t, "recall": r, "precision": p}


def _save(fig, path):
    fig.tight_layout()
    stamp(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, path, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Retained", "Churn"], yticklabels=["Retained", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    _save(fig, path)


def plot_roc(curves: dict, metrics: dict, path):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["fpr"], c["tpr"], color=PALETTE[i % 5], lw=2,
                label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (hold-out test set)"); ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_pr(curves: dict, metrics: dict, path, base_rate: float):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["recall"], c["precision"], color=PALETTE[i % 5], lw=2,
                label=f"{name} (AP={metrics[name]['pr_auc']:.3f})")
    ax.axhline(base_rate, color="k", ls="--", lw=1, label=f"No-skill ({base_rate:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (hold-out test set)")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, path)


def plot_model_comparison(table, path):
    cols = ["recall", "f1", "roc_auc", "precision", "accuracy"]
    t = table.set_index("model")[cols]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    t.plot(kind="bar", ax=ax, color=PALETTE, width=0.8)
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_xlabel("")
    ax.set_title("Model comparison on hold-out test set")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, path)


def plot_importance(df, path, title, value_col="importance", top=15):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d["feature"], d[value_col], color=PALETTE[0])
    ax.set_xlabel(value_col.replace("_", " ")); ax.set_title(title)
    _save(fig, path)
