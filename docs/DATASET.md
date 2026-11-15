# Dataset acquisition

## Real dataset (recommended): IBM Telco Customer Churn
7,043 customers, 21 columns, ~26.5% churn. Search for "IBM Telco Customer Churn" (IBM sample data
sets; a copy is also on Kaggle as `blastchar/telco-customer-churn`). Confirm the source and licence
terms yourself, then save the CSV exactly as:

    data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

Then: `python -m src.train`. The loader prefers this file whenever it exists.

Known quirks (handled by `src/preprocessing.py`): `TotalCharges` is text with blank strings for
customers with tenure 0; `SeniorCitizen` is 0/1 while other flags are Yes/No.

## Synthetic stand-in (testing only)
`python -m src.data_generation` writes `data/synthetic/telco_synthetic.csv` (same schema, simulated
churn rule, injected duplicates and blank TotalCharges). Every artifact produced from it is labelled
SYNTHETIC (dashboard banner, figure stamp, report cover). Never present those metrics as real results.

The bundled results in this repository were produced on the synthetic file because the build
environment had no internet access.
