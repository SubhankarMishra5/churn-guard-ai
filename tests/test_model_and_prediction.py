import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

from config import settings
from config.schema import RAW_FEATURES
from src import predict
from src.preprocessing import InputValidationError
from tests.helpers import sample_record, tiny_data, tiny_model


class TestModelLoading(unittest.TestCase):
    def test_missing_model_raises_with_instructions(self):
        with self.assertRaises(predict.ModelNotFoundError) as cm:
            predict.load_model(Path("/nonexistent/model.joblib"))
        self.assertIn("python -m src.train", str(cm.exception))

    def test_missing_metadata_raises(self):
        with self.assertRaises(predict.ModelNotFoundError):
            predict.load_metadata(Path("/nonexistent/meta.json"))

    def test_save_and_load_roundtrip(self):
        _, clean, _ = tiny_data()
        model = tiny_model()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.joblib"
            joblib.dump(model, p)
            loaded = predict.load_model(p)
        X = clean[RAW_FEATURES].head(20)
        np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X))

    def test_pipeline_accepts_raw_features(self):
        _, clean, _ = tiny_data()
        p = tiny_model().predict_proba(clean[RAW_FEATURES].head(10))[:, 1]
        self.assertTrue(((p >= 0) & (p <= 1)).all())


class TestSinglePrediction(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()

    def test_output_structure(self):
        r = predict.predict_single(sample_record(), self.model)
        self.assertEqual(set(r), {"probability", "prediction", "risk"})
        self.assertIn(r["prediction"], {"Churn", "Retain"})
        self.assertIn(r["risk"], {"Low", "Medium", "High"})
        self.assertTrue(0 <= r["probability"] <= 1)

    def test_risky_profile_scores_above_safe_profile(self):
        risky = sample_record(tenure=2, Contract="Month-to-month", InternetService="Fiber optic",
                              PaymentMethod="Electronic check", MonthlyCharges=95.0)
        safe = sample_record(tenure=65, Contract="Two year", PaymentMethod="Credit card (automatic)",
                             OnlineSecurity="Yes", TechSupport="Yes", MonthlyCharges=45.0)
        self.assertGreater(predict.predict_single(risky, self.model)["probability"],
                           predict.predict_single(safe, self.model)["probability"])

    def test_risk_level_boundaries(self):
        t = {"low": 0.3, "high": 0.6}
        self.assertEqual(predict.risk_level(0.29, t), "Low")
        self.assertEqual(predict.risk_level(0.30, t), "Medium")
        self.assertEqual(predict.risk_level(0.59, t), "Medium")
        self.assertEqual(predict.risk_level(0.60, t), "High")

    def test_threshold_changes_label_not_probability(self):
        a = predict.predict_single(sample_record(), self.model, threshold=0.0)
        b = predict.predict_single(sample_record(), self.model, threshold=1.01)
        self.assertEqual(a["prediction"], "Churn")
        self.assertEqual(b["prediction"], "Retain")
        self.assertEqual(a["probability"], b["probability"])

    def test_invalid_input_raises(self):
        with self.assertRaises(InputValidationError):
            predict.predict_single(sample_record(Contract="bogus"), self.model)

    def test_retention_actions(self):
        acts = predict.retention_actions(sample_record(tenure=3), "High")
        self.assertTrue(1 <= len(acts) <= 5)
        self.assertIn("Priority outreach", acts[0])
        calm = predict.retention_actions(
            sample_record(Contract="Two year", tenure=60, PaymentMethod="Credit card (automatic)",
                          InternetService="No", MonthlyCharges=20.0), "Low")
        self.assertEqual(len(calm), 1)


class TestBatchPrediction(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        _, clean, _ = tiny_data()
        self.batch = clean.drop(columns=["Churn"]).head(50)

    def test_adds_columns_keeps_order_and_length(self):
        out = predict.predict_dataframe(self.batch, self.model)
        self.assertEqual(len(out), 50)
        for c in ("ChurnProbability", "ChurnPrediction", "RiskLevel"):
            self.assertIn(c, out.columns)
        self.assertEqual(list(out["customerID"]), list(self.batch["customerID"]))

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.csv"
            self.batch.to_csv(p, index=False)
            out = predict.predict_dataframe(pd.read_csv(p), self.model)
        self.assertEqual(len(out), 50)

    def test_matches_single_prediction(self):
        row = self.batch.iloc[[0]]
        b = predict.predict_dataframe(row, self.model).iloc[0]["ChurnProbability"]
        s = predict.predict_single(row.iloc[0].to_dict(), self.model)["probability"]
        self.assertAlmostEqual(b, s, places=4)

    def test_batch_without_total_charges_works(self):
        out = predict.predict_dataframe(self.batch.drop(columns=["TotalCharges"]), self.model)
        self.assertEqual(len(out), 50)

    def test_invalid_row_fails_whole_batch_with_message(self):
        bad = self.batch.copy()
        bad.loc[bad.index[3], "Contract"] = "Daily"
        with self.assertRaises(InputValidationError):
            predict.predict_dataframe(bad, self.model)

    def test_empty_batch_rejected(self):
        with self.assertRaises(InputValidationError):
            predict.predict_dataframe(self.batch.head(0), self.model)


class TestDataLoader(unittest.TestCase):
    def test_no_dataset_gives_setup_help(self):
        from src import data_loader
        with mock.patch.object(settings, "REAL_DATA_PATH", Path("/none/a.csv")), \
             mock.patch.object(settings, "SYNTH_DATA_PATH", Path("/none/b.csv")):
            with self.assertRaises(data_loader.DatasetNotFoundError) as cm:
                data_loader.resolve_data_path()
        self.assertIn("--generate-synthetic", str(cm.exception))

    def test_schema_error_lists_missing_columns(self):
        from src import data_loader
        with self.assertRaises(data_loader.SchemaError) as cm:
            data_loader.validate_raw_schema(pd.DataFrame({"tenure": [1]}))
        self.assertIn("Churn", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
