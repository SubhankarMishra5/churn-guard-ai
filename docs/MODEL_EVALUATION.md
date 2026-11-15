# Model Evaluation

> **SYNTHETIC DATA WARNING.** These numbers were produced on a simulated dataset (same schema as IBM Telco) for software testing. They are **not** real-world results. Run `python -m src.train` with the real CSV and `python scripts/build_all.py` to regenerate this file.

- Trained: 2026-11-14T09:57:10+00:00
- Data source: **synthetic**
- Rows: 7,043 (train 5,634 / test 1,409, stratified)
- Churn rate: 26.3%; imbalance handled by: class weights (balanced) / scale_pos_weight; no resampling
- Tuning: randomised search, 10 configs, stratified 5-fold CV, refit on F1
- Selection rule: highest mean of CV recall, F1, ROC-AUC -> **Logistic Regression**
- XGBoost installed during training: False (scikit-learn HistGradientBoosting used as fallback)
- Explainer backend: Baseline-substitution attribution (SHAP not installed) (probability points)

## Model comparison

| Model | CV recall | CV F1 | CV ROC-AUC | Test recall | Test F1 | Test ROC-AUC | Test precision | Test accuracy |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.795 | 0.616 | 0.833 | 0.784 | 0.627 | 0.840 | 0.522 | 0.754 |
| Gradient Boosting (XGBoost fallback) | 0.792 | 0.616 | 0.826 | 0.779 | 0.624 | 0.833 | 0.521 | 0.753 |
| Random Forest | 0.765 | 0.615 | 0.827 | 0.776 | 0.630 | 0.835 | 0.530 | 0.760 |
| Decision Tree | 0.780 | 0.601 | 0.808 | 0.741 | 0.597 | 0.816 | 0.500 | 0.737 |

## Selected model: Logistic Regression

Confusion matrix (test): TN=772, FP=266, FN=80, TP=291

```
              precision    recall  f1-score   support

    Retained       0.91      0.74      0.82      1038
       Churn       0.52      0.78      0.63       371

    accuracy                           0.75      1409
   macro avg       0.71      0.76      0.72      1409
weighted avg       0.81      0.75      0.77      1409
```

Best hyper-parameters:

```json
{
  "Logistic Regression": {
    "C": 3
  },
  "Decision Tree": {
    "min_samples_leaf": 80,
    "max_depth": 8,
    "criterion": "gini"
  },
  "Random Forest": {
    "n_estimators": 150,
    "min_samples_leaf": 15,
    "max_features": "sqrt",
    "max_depth": 8
  },
  "Gradient Boosting (XGBoost fallback)": {
    "min_samples_leaf": 20,
    "max_iter": 100,
    "max_depth": 4,
    "learning_rate": 0.03,
    "l2_regularization": 0.0
  }
}
```

## Top features (permutation importance, drop in ROC-AUC)

| Feature | Importance |
|---|---|
| Tenure (months) | 0.1917 |
| Contract type | 0.1184 |
| Internet service | 0.0544 |
| Total charges | 0.0273 |
| Payment method | 0.0109 |
| Online security | 0.0060 |
| Tech support | 0.0044 |
| Paperless billing | 0.0039 |

Figures: `reports/figures/` (confusion matrix, ROC, precision-recall, comparison, importance, EDA). Interpretation limits: differences between top models are small relative to fold-to-fold variation; probabilities are not calibrated; the 0.50 threshold is a default, not cost-optimised.
