"""Generate assets/architecture.png (system architecture diagram)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import settings

NAVY, ORANGE, GREEN, GREY = "#1F4E79", "#E4572E", "#2E933C", "#EEF2F7"


def box(ax, x, y, w, h, title, sub="", fc=NAVY, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2 + (0.14 if sub else 0), title, ha="center", va="center",
            color=tc, fontsize=9.5, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", color=tc, fontsize=7.5)


def arrow(ax, p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.4,
                                 color="#444", connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(6, 5.95, "ChurnGuard AI - System Architecture", ha="center", fontsize=14,
            fontweight="bold", color=NAVY)

    ax.add_patch(FancyBboxPatch((0.15, 2.55), 11.7, 2.85, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=GREY, ec="#C9D3E0"))
    ax.text(0.35, 5.2, "TRAINING PIPELINE  (python -m src.train)", fontsize=8.5, color="#555",
            fontweight="bold")
    xs = [0.4, 2.55, 4.7, 6.85, 9.0]
    w, h, y = 2.0, 1.15, 3.75
    box(ax, xs[0], y, w, h, "Data source", "IBM Telco CSV\n(or synthetic test data)", GREEN)
    box(ax, xs[1], y, w, h, "Validate & clean", "schema check, dedupe,\nmissing values, target")
    box(ax, xs[2], y, w, h, "Feature engineering", "tenure groups, add-on\ncount, charge ratios")
    box(ax, xs[3], y, w, h, "Preprocessing", "StandardScaler +\nOne-Hot (in Pipeline)")
    box(ax, xs[4], y, w, h, "Model training", "LR, DT, RF, XGBoost\n5-fold CV + tuning")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, y + h / 2), (b, y + h / 2))
    box(ax, 9.0, 2.75, w, 0.8, "Evaluation & selection", "recall, F1, ROC-AUC", ORANGE)
    arrow(ax, (10.0, y), (10.0, 3.55))
    box(ax, 4.7, 2.75, 3.9, 0.8, "Persist artifacts",
        "models/best_model.joblib + metadata.json + reports/", ORANGE)
    arrow(ax, (9.0, 3.15), (8.6, 3.15))

    box(ax, 0.4, 0.35, 2.4, 1.2, "predict.py", "validation, probability,\nrisk band, retention rules")
    box(ax, 3.2, 0.35, 2.4, 1.2, "explainability.py", "SHAP or baseline-\nsubstitution attribution")
    box(ax, 6.0, 0.35, 2.4, 1.2, "analytics.py", "churn by tenure, contract,\npayment, charges, segments")
    box(ax, 8.8, 0.35, 3.0, 1.2, "Streamlit app (app.py)",
        "Dashboard | Prediction | Analytics\nModel Performance | Batch CSV", NAVY)
    ax.text(8.8, 1.8, "SERVING LAYER  (streamlit run app.py)", fontsize=8.5, color="#555",
            fontweight="bold")
    arrow(ax, (6.65, 2.75), (1.6, 1.58), rad=0.12)
    arrow(ax, (2.8, 0.95), (3.2, 0.95)); arrow(ax, (5.6, 0.95), (6.0, 0.95))
    arrow(ax, (8.4, 0.95), (8.8, 0.95))
    ax.text(6.0, 0.08, "Users: analysts upload CSVs or enter a customer; the app returns probability, "
            "risk level, drivers and retention actions.", ha="center", fontsize=8, color="#444")
    settings.ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(settings.ASSETS_DIR / "architecture.png", dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    main()
