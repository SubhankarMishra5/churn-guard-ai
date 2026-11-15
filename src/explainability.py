"""Explainable AI: SHAP when available, model-agnostic attribution otherwise.

Backends
--------
* ``shap``      - TreeExplainer (RandomForest / DecisionTree / XGBoost) or
                  LinearExplainer (Logistic Regression) on the transformed matrix;
                  one-hot columns are summed back to the original feature.
* ``baseline``  - single-feature substitution: for each raw feature, replace the
                  customer's value with values drawn from a background sample and
                  measure the average drop/rise in churn probability.
                  Works for any model and needs no extra packages.

Both return one contribution per *original* feature (positive = raises churn risk).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.schema import FRIENDLY, RAW_FEATURES

try:                                    # optional dependency
    import shap
    SHAP_AVAILABLE = True
except Exception:                       # pragma: no cover - depends on environment
    shap = None
    SHAP_AVAILABLE = False


class ChurnExplainer:
    def __init__(self, pipeline, background: pd.DataFrame, prefer_shap: bool = True,
                 n_background: int = 60, seed: int = 0):
        self.pipeline = pipeline
        self.background = background[RAW_FEATURES].reset_index(drop=True)
        self.bg_small = self.background.sample(min(n_background, len(self.background)),
                                               random_state=seed).reset_index(drop=True)
        self.backend, self.unit, self._shap = "baseline", "probability points", None
        if prefer_shap and SHAP_AVAILABLE:
            try:
                self._init_shap()
            except Exception:           # fall back silently to the baseline method
                self.backend, self._shap = "baseline", None

    # ------------------------------------------------------------- SHAP path
    def _init_shap(self):
        clf = self.pipeline.named_steps["clf"]
        prep = self.pipeline[:-1]
        self._names = list(self.pipeline.named_steps["prep"].get_feature_names_out())
        self._groups = [self._group_of(n) for n in self._names]
        kind = type(clf).__name__
        if kind in ("RandomForestClassifier", "DecisionTreeClassifier"):
            self._shap, self.unit = shap.TreeExplainer(clf), "probability points"
        elif kind == "XGBClassifier":
            self._shap, self.unit = shap.TreeExplainer(clf), "log-odds"
        elif kind == "LogisticRegression":
            bg_t = np.asarray(prep.transform(self.bg_small))
            self._shap, self.unit = shap.LinearExplainer(clf, bg_t), "log-odds"
        else:
            raise ValueError("model type not supported by SHAP path")
        self._prep = prep
        self.backend = "shap"

    @staticmethod
    def _group_of(name: str) -> str:
        body = name.split("__", 1)[1]
        for raw in sorted(FRIENDLY, key=len, reverse=True):
            if body == raw or body.startswith(raw + "_"):
                return raw
        return body

    def _shap_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = np.asarray(self._prep.transform(X))
        sv = self._shap.shap_values(Xt)
        if isinstance(sv, list):                 # older shap: [class0, class1]
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                         # newer shap: (n, features, classes)
            sv = sv[:, :, 1]
        df = pd.DataFrame(sv, columns=self._groups)
        return df.T.groupby(level=0).sum().T

    # ------------------------------------------------------ baseline path
    def _baseline_row(self, row: pd.DataFrame) -> np.ndarray:
        n = len(self.bg_small)
        p0 = float(self.pipeline.predict_proba(row)[0, 1])
        rep = row.loc[row.index.repeat(n)].reset_index(drop=True)
        blocks = []
        for f in RAW_FEATURES:
            blk = rep.copy()
            blk[f] = self.bg_small[f].values
            blocks.append(blk)
        probs = self.pipeline.predict_proba(pd.concat(blocks, ignore_index=True))[:, 1]
        return p0 - probs.reshape(len(RAW_FEATURES), n).mean(axis=1)

    def _baseline_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        rows = [self._baseline_row(X.iloc[[i]]) for i in range(len(X))]
        return pd.DataFrame(np.vstack(rows), columns=RAW_FEATURES)

    # ------------------------------------------------------------- public
    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[RAW_FEATURES].reset_index(drop=True)
        return self._shap_matrix(X) if self.backend == "shap" else self._baseline_matrix(X)

    def local(self, record: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contributions for a single customer (1-row DataFrame)."""
        c = self.contributions(record.iloc[[0]]).iloc[0]
        out = pd.DataFrame({
            "feature": c.index,
            "label": [FRIENDLY.get(f, f) for f in c.index],
            "value": [record.iloc[0][f] for f in c.index],
            "contribution": c.values})
        out["direction"] = np.where(out["contribution"] >= 0, "raises risk", "lowers risk")
        return out.reindex(out["contribution"].abs().sort_values(ascending=False).index
                           ).reset_index(drop=True)

    def global_importance(self, X: pd.DataFrame, n_rows: int = 80, seed: int = 0) -> pd.DataFrame:
        X = X.sample(min(n_rows, len(X)), random_state=seed)
        c = self.contributions(X).abs().mean().sort_values(ascending=False)
        return pd.DataFrame({"feature": [FRIENDLY.get(f, f) for f in c.index],
                             "importance": c.values})

    @property
    def backend_label(self) -> str:
        if self.backend == "shap":
            return "SHAP"
        note = "" if SHAP_AVAILABLE else " (SHAP not installed)"
        return f"Baseline-substitution attribution{note}"


def explanation_text(contrib: pd.DataFrame, top: int = 3) -> list[str]:
    """Plain-English sentences for the strongest drivers."""
    lines = []
    for _, r in contrib.head(top).iterrows():
        v = r["value"]
        v = f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
        verb = "increases" if r["contribution"] >= 0 else "reduces"
        lines.append(f"{r['label']} = {v} {verb} this customer's churn risk.")
    return lines
