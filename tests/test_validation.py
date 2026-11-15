import unittest

import pandas as pd

from src.preprocessing import InputValidationError, validate_input_frame
from tests.helpers import sample_record


class TestInputValidation(unittest.TestCase):
    def test_valid_record_passes_and_estimates_total(self):
        out = validate_input_frame(pd.DataFrame([sample_record()]))
        self.assertAlmostEqual(out["TotalCharges"].iloc[0], 12 * 50.0)

    def test_explicit_total_charges_kept(self):
        out = validate_input_frame(pd.DataFrame([sample_record(TotalCharges=999.0)]))
        self.assertEqual(out["TotalCharges"].iloc[0], 999.0)

    def test_missing_columns_rejected(self):
        rec = sample_record(); rec.pop("Contract")
        with self.assertRaises(InputValidationError) as cm:
            validate_input_frame(pd.DataFrame([rec]))
        self.assertIn("Contract", str(cm.exception))

    def test_invalid_category_rejected(self):
        with self.assertRaises(InputValidationError) as cm:
            validate_input_frame(pd.DataFrame([sample_record(Contract="Weekly")]))
        self.assertIn("Contract", cm.exception.problems[0])

    def test_negative_tenure_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_input_frame(pd.DataFrame([sample_record(tenure=-3)]))

    def test_non_numeric_charges_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_input_frame(pd.DataFrame([sample_record(MonthlyCharges="abc")]))

    def test_bad_senior_flag_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_input_frame(pd.DataFrame([sample_record(SeniorCitizen=2)]))

    def test_empty_frame_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_input_frame(pd.DataFrame())

    def test_whitespace_in_categories_is_trimmed(self):
        out = validate_input_frame(pd.DataFrame([sample_record(Contract=" One year ")]))
        self.assertEqual(out["Contract"].iloc[0], "One year")

    def test_multiple_problems_reported_together(self):
        with self.assertRaises(InputValidationError) as cm:
            validate_input_frame(pd.DataFrame([sample_record(tenure=-1, Contract="x")]))
        self.assertGreaterEqual(len(cm.exception.problems), 2)


if __name__ == "__main__":
    unittest.main()
