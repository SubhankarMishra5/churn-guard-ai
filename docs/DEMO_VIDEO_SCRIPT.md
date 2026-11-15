# Demonstration Video Script (5-7 minutes)

Record your own screen; nothing here is pre-recorded. Values shown come from the bundled
synthetic-trained model - yours will differ if you retrain on real data.

| Time | Screen | Narration |
|---|---|---|
| 0:00-0:30 | Title slide, then terminal in project folder | "This is ChurnGuard AI. It predicts which telecom customers are likely to churn, explains why, and recommends retention actions. Today's demo uses a synthetic dataset with the IBM Telco schema; switch to the real CSV by dropping it in data/raw." (Say this only if you are on synthetic data.) |
| 0:30-1:30 | Terminal: `python -m src.data_generation` (or show the real CSV in data/raw), then open `reports/figures/eda_*.png` | "First the data: 7,058 raw rows. Cleaning removes 15 duplicates and repairs 11 blank TotalCharges, leaving 7,043 customers; 26% churn, so classes are imbalanced. Contract type, tenure and payment method visibly separate churners." |
| 1:30-2:45 | Terminal: `python -m src.train` (speed up the wait in editing) | "Training tunes four models with stratified 5-fold cross-validation. We rank by the mean of recall, F1 and ROC-AUC because missing a churner is costly. Here Logistic Regression wins narrowly; models are within noise." |
| 2:45-3:30 | `streamlit run app.py` -> Dashboard page | "KPIs: total customers, predicted churners, churn percentage, high-risk count, and hold-out recall, F1, AUC. The yellow banner reminds us this is synthetic data." |
| 3:30-4:30 | Customer Prediction page | Enter: tenure 3, Month-to-month, Fiber optic, Electronic check, monthly 95.50, no security/tech support. "Predicted churn, High risk (above 90% in the bundled model). Top drivers: short tenure, month-to-month, fiber. Retention actions: priority outreach, discounted annual plan, auto-pay credit, free tech-support bundle." Then enter a 60-month, Two-year, credit-card customer: "Low risk (about 3%)." |
| 4:30-5:15 | Analytics page | "Churn by tenure group, contract, payment method, monthly-charge band and segment. Use the filters to focus on fiber or month-to-month customers." |
| 5:15-5:50 | Model Performance page | "Comparison table, confusion matrix, ROC and PR curves, permutation importance and the explanation method in use." |
| 5:50-6:30 | Batch Prediction page | Download the template, upload it, show the scored table, click Download. Then upload a file with a bad Contract value to show the validation message. |
| 6:30-7:00 | Terminal: `python -m unittest discover -s tests -t .` | "52 automated tests pass. Next steps: train on the real data, calibrate probabilities, optimise the threshold by cost, and deploy behind an API. Thank you." |

Tips: record at 1080p, zoom the browser to 110%, keep narration to ~130 words/minute.
