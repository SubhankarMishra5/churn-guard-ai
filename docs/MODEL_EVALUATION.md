# Model Evaluation

- Trained: 2026-11-15T09:27:49+00:00
- Data source: **real**
- Rows: 7,043 (train 5,634 / test 1,409, stratified)
- Churn rate: 26.5%; imbalance handled by: class weights (balanced) / scale_pos_weight; no resampling
- Tuning: randomised search, 10 configs, stratified 5-fold CV, refit on F1
- Selection rule: highest mean of CV recall, F1, ROC-AUC -> **Random Forest**
- XGBoost installed during training: True
- Explainer backend: SHAP (probability points)

## Model comparison

| Model | CV recall | CV F1 | CV ROC-AUC | Test recall | Test F1 | Test ROC-AUC | Test precision | Test accuracy |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.790 | 0.634 | 0.847 | 0.775 | 0.630 | 0.844 | 0.531 | 0.759 |
| Logistic Regression | 0.790 | 0.629 | 0.846 | 0.791 | 0.624 | 0.840 | 0.515 | 0.747 |
| XGBoost | 0.765 | 0.637 | 0.844 | 0.770 | 0.627 | 0.839 | 0.528 | 0.757 |
| Decision Tree | 0.785 | 0.610 | 0.828 | 0.794 | 0.631 | 0.832 | 0.523 | 0.753 |

## Selected model: Random Forest

Confusion matrix (test): TN=779, FP=256, FN=84, TP=290

```
              precision    recall  f1-score   support

    Retained       0.90      0.75      0.82      1035
       Churn       0.53      0.78      0.63       374

    accuracy                           0.76      1409
   macro avg       0.72      0.76      0.73      1409
weighted avg       0.80      0.76      0.77      1409
```

Best hyper-parameters:

```json
{
  "Logistic Regression": {
    "C": 0.01
  },
  "Decision Tree": {
    "min_samples_leaf": 80,
    "max_depth": 6,
    "criterion": "gini"
  },
  "Random Forest": {
    "n_estimators": 150,
    "min_samples_leaf": 8,
    "max_features": 0.3,
    "max_depth": 6
  },
  "XGBoost": {
    "subsample": 0.8,
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.1,
    "colsample_bytree": 0.7
  }
}
```

## Top features (permutation importance, drop in ROC-AUC)

| Feature | Importance |
|---|---|
| Contract type | 0.1038 |
| Tenure (months) | 0.0400 |
| Internet service | 0.0256 |
| Total charges | 0.0091 |
| Online security | 0.0064 |
| Tech support | 0.0044 |
| Monthly charges | 0.0025 |
| Payment method | 0.0020 |

Figures: `reports/figures/` (confusion matrix, ROC, precision-recall, comparison, importance, EDA). Interpretation limits: differences between top models are small relative to fold-to-fold variation; probabilities are not calibrated; the 0.50 threshold is a default, not cost-optimised.
