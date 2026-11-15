# ChurnGuard AI - Slide-by-Slide Speaking Script

Approximate total time: 8-10 minutes (about 45-60 seconds per slide).

## Slide 1: Title

Good morning. I'm presenting ChurnGuard AI, an end-to-end system that predicts which telecom customers are likely to leave, explains why, and suggests how to keep them. One honesty point up front: the numbers you'll see today were produced on a synthetic dataset that mirrors the IBM Telco schema, because I had no internet access to download the real file during the build. They prove the pipeline works; they are not claims about real customers. Re-running one command on the real CSV regenerates everything.

## Slide 2: The problem: customers leave, and it is costly

Churn means a customer cancels their service. Winning a new customer usually costs more than keeping an existing one, so companies want to find at-risk customers early. The challenge is that churners are a minority - about one in four here - so a model that simply says nobody will leave looks 74 percent accurate and is useless. That is why we focus on recall and F1 alongside ROC-AUC, not accuracy alone.

## Slide 3: Objectives and technology stack

The project has five goals: a reproducible preprocessing pipeline, four compared models with cross-validation and tuning, proper evaluation, explanations for each prediction, and a usable dashboard backed by automated tests. The stack is standard Python data science tooling with Streamlit for the interface.

## Slide 4: Dataset and preprocessing

The intended dataset is IBM Telco Customer Churn with about seven thousand customers. In this run I used a synthetic dataset with the same columns, deliberately including duplicates and blank total-charge values so the cleaning code is exercised. Cleaning removed 15 duplicates and repaired 11 blank TotalCharges values, leaving 7,043 customers. I engineered features such as add-on service count, month-to-month flag, auto-pay flag and tenure group. Scaling and one-hot encoding live inside the sklearn pipeline so no information leaks from the test set. Imbalance is handled with class weights rather than resampling.

## Slide 5: System architecture

On the top is the training pipeline: data, validation and cleaning, feature engineering, preprocessing, model training with cross-validation, then evaluation, selection and persistence of the best model plus metadata. Below is the serving layer: prediction, explainability and analytics modules feeding the Streamlit app. Because the whole pipeline is saved as one object, prediction applies exactly the same steps as training.

## Slide 6: Models and methodology

Four models were tuned with randomised search and stratified five-fold cross-validation, refit on F1. I selected the winner by the average of cross-validated recall, F1 and ROC-AUC rather than test performance, so the test set stays an honest final check. The gradient boosting entry is scikit-learn's HistGradientBoosting because XGBoost could not be installed offline; the code uses real XGBoost automatically when it is available. Logistic Regression ranked first, but the models are close - differences are within fold-to-fold noise.

## Slide 7: Results: model comparison

On the held-out test set, Logistic Regression reached recall 0.784, F1 0.627 and ROC-AUC 0.840. The bar chart shows all four models are close. Because the data is synthetic and its churn rule is logistic, a logistic model being best is expected, and this ranking should not be generalised to real data.

## Slide 8: Evaluation of the selected model

The confusion matrix shows 291 churners caught and 80 missed, with 266 false alarms. Class weighting deliberately favours recall, so precision is about 52%. Whether that trade is acceptable depends on the cost of a retention offer versus the value of a saved customer, and the threshold can be adjusted. The ROC curves show all models clearly above chance.

## Slide 9: Explainable AI and the dashboard

Each prediction is explained by per-feature contributions. This run used the Baseline-substitution attribution (SHAP not installed) method; the SHAP code path is implemented but was not executed here because SHAP could not be installed offline. The chart shows the global drivers. The Streamlit dashboard has five pages: Dashboard, Customer Prediction with risk level, reasons and retention actions, interactive Analytics, Model Performance, and Batch Prediction with CSV upload and download.

## Slide 10: Conclusion, limitations and future work

To summarise: a complete, tested churn workflow with four compared models, evaluation centred on recall, F1 and ROC-AUC, explanations and a dashboard. The main limitation is that results are on synthetic data, so the first next step is to train on the real IBM Telco file and regenerate the report. Other future work: cost-sensitive thresholds, probability calibration, uplift modelling, fairness auditing and deployment behind an API. Thank you - happy to take questions.
