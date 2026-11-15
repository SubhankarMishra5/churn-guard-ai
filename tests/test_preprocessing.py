import unittest

import numpy as np
import pandas as pd

from config.schema import RAW_FEATURES
from src.feature_engineering import add_features
from src.preprocessing import (build_preprocessor, class_balance, clean_data,
                               outlier_report)
from tests.helpers import tiny_data


class TestCleaning(unittest.TestCase):
    def setUp(self):
        self.raw, self.clean, self.report = tiny_data()

    def test_duplicates_removed(self):
        self.assertEqual(self.report["duplicates_removed"], 5)
        self.assertEqual(len(self.clean), len(self.raw) - 5)
        self.assertFalse(self.clean.duplicated().any())

    def test_blank_total_charges_fixed(self):
        self.assertEqual(self.report["total_charges_blank"], 11)
        self.assertEqual(self.clean["TotalCharges"].isna().sum(), 0)
        zero_tenure = self.clean[self.clean["tenure"] == 0]
        self.assertTrue((zero_tenure["TotalCharges"] == 0).all())

    def test_target_is_binary_int(self):
        self.assertEqual(set(self.clean["Churn"].unique()), {0, 1})

    def test_no_missing_in_features(self):
        self.assertEqual(self.clean[RAW_FEATURES].isna().sum().sum(), 0)

    def test_bad_target_rows_dropped(self):
        raw = self.raw.copy()
        raw.loc[0, "Churn"] = "maybe"
        _, rep = clean_data(raw)
        self.assertEqual(rep["rows_dropped_bad_target"], 1)

    def test_outlier_report_structure(self):
        rep = outlier_report(self.clean)
        self.assertEqual(set(rep), {"tenure", "MonthlyCharges", "TotalCharges"})
        self.assertTrue(all(v["n_outliers"] >= 0 for v in rep.values()))

    def test_class_balance(self):
        b = class_balance(pd.Series([0, 0, 0, 1]))
        self.assertEqual(b["n_churn"], 1)
        self.assertAlmostEqual(b["churn_rate"], 0.25)
        self.assertAlmostEqual(b["imbalance_ratio"], 3.0)


class TestFeatureEngineering(unittest.TestCase):
    def test_new_columns_and_no_nan(self):
        _, clean, _ = tiny_data()
        f = add_features(clean[RAW_FEATURES])
        for c in ["NumAddOnServices", "AvgMonthlySpend", "ChargeRatio", "TenureGroup",
                  "HasInternet", "IsMonthToMonth", "AutoPay", "NoProtection"]:
            self.assertIn(c, f.columns)
        self.assertEqual(f[["AvgMonthlySpend", "ChargeRatio"]].isna().sum().sum(), 0)
        self.assertTrue(np.isfinite(f["ChargeRatio"]).all())

    def test_tenure_zero_is_safe(self):
        _, clean, _ = tiny_data()
        row = clean[RAW_FEATURES].head(1).copy()
        row["tenure"], row["TotalCharges"] = 0, 0.0
        out = add_features(row)
        self.assertEqual(out["ChargeRatio"].iloc[0], 1.0)

    def test_input_not_mutated(self):
        _, clean, _ = tiny_data()
        X = clean[RAW_FEATURES]
        cols = list(X.columns)
        add_features(X)
        self.assertEqual(list(X.columns), cols)


class TestPreprocessor(unittest.TestCase):
    def test_output_is_numeric_and_scaled(self):
        _, clean, _ = tiny_data()
        prep = build_preprocessor()
        Xt = prep.fit_transform(add_features(clean[RAW_FEATURES]))
        self.assertTrue(np.isfinite(Xt).all())
        names = list(prep.get_feature_names_out())
        self.assertAlmostEqual(float(Xt[:, names.index("num__tenure")].mean()), 0, places=6)

    def test_unknown_category_does_not_crash(self):
        _, clean, _ = tiny_data()
        prep = build_preprocessor().fit(add_features(clean[RAW_FEATURES]))
        row = add_features(clean[RAW_FEATURES].head(1).copy())
        row["Contract"] = "Five year"
        self.assertEqual(prep.transform(row).shape[0], 1)


if __name__ == "__main__":
    unittest.main()
