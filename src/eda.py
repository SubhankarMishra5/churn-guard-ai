"""Exploratory data analysis figures (saved to reports/figures)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import settings
from src.evaluation import PALETTE, stamp

T = settings.TARGET


def _save(fig, name):
    settings.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    stamp(fig)
    fig.savefig(settings.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(df: pd.DataFrame) -> list[str]:
    d = df.copy()
    d["ChurnLabel"] = d[T].map({1: "Churn", 0: "Retained"})
    cmap = {"Churn": PALETTE[1], "Retained": PALETTE[0]}
    made = []

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    counts = d["ChurnLabel"].value_counts()
    ax.bar(counts.index, counts.values, color=[cmap[i] for i in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({100 * v / len(d):.1f}%)", ha="center", va="bottom")
    ax.set_title("Class distribution"); ax.set_ylabel("Customers")
    _save(fig, "eda_class_distribution.png"); made.append("eda_class_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(d, x="tenure", hue="ChurnLabel", palette=cmap, bins=24,
                 multiple="layer", ax=axes[0])
    axes[0].set_title("Tenure distribution")
    sns.boxplot(d, x="ChurnLabel", y="MonthlyCharges", hue="ChurnLabel",
                palette=cmap, legend=False, ax=axes[1])
    axes[1].set_title("Monthly charges by churn")
    _save(fig, "eda_tenure_charges.png"); made.append("eda_tenure_charges.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["Contract", "PaymentMethod"]):
        rate = d.groupby(col)[T].mean().mul(100).sort_values()
        ax.barh(rate.index, rate.values, color=PALETTE[0])
        ax.set_xlabel("Churn rate (%)"); ax.set_title(f"Churn rate by {col}")
    _save(fig, "eda_churn_by_category.png"); made.append("eda_churn_by_category.png")

    num = d[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", T]]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation (numeric features)")
    _save(fig, "eda_correlation.png"); made.append("eda_correlation.png")
    return made
