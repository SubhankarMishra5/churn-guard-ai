import unittest

import pandas as pd

from config.schema import RAW_FEATURES
from src import analytics
from src.explainability import ChurnExplainer, explanation_text
from src.preprocessing import validate_input_frame
from tests.helpers import sample_record, tiny_data, tiny_model


class TestExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, clean, _ = tiny_data()
        cls.expl = ChurnExplainer(tiny_model(), clean[RAW_FEATURES].head(200),
                                  prefer_shap=False)
        cls.clean = clean

    def test_local_covers_all_features_sorted(self):
        rec = validate_input_frame(pd.DataFrame([sample_record(tenure=2)]))
        loc = self.expl.local(rec)
        self.assertEqual(set(loc["feature"]), set(RAW_FEATURES))
        a = loc["contribution"].abs().tolist()
        self.assertEqual(a, sorted(a, reverse=True))

    def test_month_to_month_pushes_risk_up(self):
        rec = validate_input_frame(pd.DataFrame([sample_record(Contract="Month-to-month")]))
        loc = self.expl.local(rec).set_index("feature")
        self.assertGreater(loc.loc["Contract", "contribution"], 0)

    def test_global_importance(self):
        g = self.expl.global_importance(self.clean[RAW_FEATURES], n_rows=20)
        self.assertEqual(len(g), len(RAW_FEATURES))
        self.assertTrue((g["importance"] >= 0).all())

    def test_text_explanations(self):
        rec = validate_input_frame(pd.DataFrame([sample_record()]))
        lines = explanation_text(self.expl.local(rec), top=3)
        self.assertEqual(len(lines), 3)
        self.assertTrue(all("churn risk" in s for s in lines))

    def test_backend_label_reports_method(self):
        self.assertIn(self.expl.backend, {"baseline", "shap"})


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        _, self.df, _ = tiny_data()

    def test_tables_consistent(self):
        for tbl in (analytics.tenure_group(self.df), analytics.charge_band(self.df),
                    analytics.churn_rate_by(self.df, "Contract"),
                    analytics.segment_table(self.df)):
            self.assertEqual(int(tbl["customers"].sum()), len(self.df))
            self.assertTrue(tbl["churn_rate"].between(0, 100).all())

    def test_kpis(self):
        scored = pd.DataFrame({"ChurnPrediction": [1, 0, 1, 0],
                               "RiskLevel": ["High", "Low", "Medium", "Low"]})
        k = analytics.kpis(scored)
        self.assertEqual((k["total"], k["predicted_churn"], k["high_risk"]), (4, 2, 1))
        self.assertEqual(k["predicted_churn_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
