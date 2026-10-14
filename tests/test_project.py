"""Automated Unit and Integration Tests for Employee Attrition Prediction System.

Tests dataset generation, schema integrity, preprocessing pipelines, model
training/saving/loading, inference probability boundaries, and input validation.
"""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.generate_data import generate_synthetic_data, save_dataset
from src.predict import AttritionPredictor, predict_employee_attrition
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    clean_and_validate_data,
    create_preprocessor,
    prepare_train_test_data,
)
from src.train_model import train_and_evaluate_models
from src.utils import (
    ALL_DATA_COLUMNS,
    MODEL_FILE,
    DataError,
    ModelError,
    ValidationError,
    validate_dataset_schema,
    validate_employee_input,
)


class TestDatasetGeneration(unittest.TestCase):
    """Tests for synthetic dataset generator and schema requirements."""

    def test_synthetic_data_generation_shape_and_columns(self):
        df = generate_synthetic_data(n_samples=100, random_seed=42)
        self.assertEqual(len(df), 100)
        for col in ALL_DATA_COLUMNS:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    def test_target_labels(self):
        df = generate_synthetic_data(n_samples=100, random_seed=42)
        unique_targets = set(df["Attrition"].unique())
        self.assertTrue(unique_targets.issubset({"Yes", "No"}))
        self.assertIn("Yes", unique_targets)
        self.assertIn("No", unique_targets)

    def test_reproducibility(self):
        df1 = generate_synthetic_data(n_samples=50, random_seed=123)
        df2 = generate_synthetic_data(n_samples=50, random_seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_domain_constraints_in_generated_data(self):
        df = generate_synthetic_data(n_samples=100, random_seed=42)
        self.assertTrue((df["Age"] >= 18).all())
        self.assertTrue((df["MonthlyIncome"] > 0).all())
        self.assertTrue((df["YearsAtCompany"] >= 0).all())
        # Tenure cannot exceed age - 18
        self.assertTrue((df["YearsAtCompany"] <= (df["Age"] - 18)).all())
        self.assertTrue(df["JobSatisfaction"].isin([1, 2, 3, 4, 5]).all())
        self.assertTrue(df["WorkLifeBalance"].isin([1, 2, 3, 4, 5]).all())
        self.assertTrue(df["PerformanceRating"].isin([1, 2, 3, 4, 5]).all())


class TestPreprocessing(unittest.TestCase):
    """Tests for preprocessing transformations and data hygiene."""

    def setUp(self):
        self.df = generate_synthetic_data(n_samples=120, random_seed=42)

    def test_clean_and_validate_removes_duplicates(self):
        # Create duplicate row
        df_with_dups = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        self.assertEqual(len(df_with_dups), 121)
        cleaned_df, _ = clean_and_validate_data(df_with_dups, drop_duplicates=True)
        self.assertEqual(len(cleaned_df), 120)

    def test_exclusion_of_employee_id(self):
        preprocessor = create_preprocessor()
        X = self.df[FEATURE_COLUMNS].copy()
        # Ensure EmployeeID is NOT in input features
        self.assertNotIn("EmployeeID", X.columns)
        # Verify preprocessor transforms without needing EmployeeID
        transformed = preprocessor.fit_transform(X)
        self.assertIsInstance(transformed, np.ndarray)
        # Should have numerical features + one-hot encoded categories
        self.assertGreater(transformed.shape[1], len(NUMERICAL_FEATURES))

    def test_stratified_split_preserves_distribution(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self.df.to_csv(tmp_path, index=False)
            X_train, X_test, y_train, y_test = prepare_train_test_data(
                file_path=tmp_path, test_size=0.20, random_seed=42
            )
            self.assertEqual(len(X_train), 96)
            self.assertEqual(len(X_test), 24)
            # Check target labels are 0 and 1 integers
            self.assertTrue(set(y_train.unique()).issubset({0, 1}))
            self.assertTrue(set(y_test.unique()).issubset({0, 1}))
            # Stratification check (similar ratio)
            train_ratio = y_train.mean()
            test_ratio = y_test.mean()
            self.assertAlmostEqual(train_ratio, test_ratio, delta=0.10)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestInputValidation(unittest.TestCase):
    """Tests for boundary checks and error handling utilities."""

    def setUp(self):
        self.valid_record = {
            "Age": 32,
            "Department": "Engineering",
            "MonthlyIncome": 65000.0,
            "YearsAtCompany": 4,
            "JobSatisfaction": 4,
            "WorkLifeBalance": 3,
            "PerformanceRating": 4,
            "OverTime": "No",
            "DistanceFromHome": 10.0,
        }

    def test_valid_input_passes(self):
        validated = validate_employee_input(self.valid_record)
        self.assertEqual(validated["Age"], 32)
        self.assertEqual(validated["Department"], "Engineering")

    def test_reject_missing_fields(self):
        incomplete = self.valid_record.copy()
        del incomplete["Department"]
        with self.assertRaises(ValidationError) as ctx:
            validate_employee_input(incomplete)
        self.assertIn("Missing required fields", str(ctx.exception))

    def test_reject_invalid_department(self):
        bad_dept = self.valid_record.copy()
        bad_dept["Department"] = "Aerospace"
        with self.assertRaises(ValidationError) as ctx:
            validate_employee_input(bad_dept)
        self.assertIn("Department must be one of", str(ctx.exception))

    def test_reject_negative_or_extreme_age(self):
        bad_age = self.valid_record.copy()
        bad_age["Age"] = -5
        with self.assertRaises(ValidationError):
            validate_employee_input(bad_age)

        bad_age["Age"] = 99
        with self.assertRaises(ValidationError):
            validate_employee_input(bad_age)

    def test_reject_negative_income(self):
        bad_income = self.valid_record.copy()
        bad_income["MonthlyIncome"] = -1000.0
        with self.assertRaises(ValidationError):
            validate_employee_input(bad_income)

    def test_reject_impossible_tenure(self):
        bad_tenure = self.valid_record.copy()
        bad_tenure["Age"] = 22
        bad_tenure["YearsAtCompany"] = 15  # Impossible since 22 - 18 = 4 max
        with self.assertRaises(ValidationError) as ctx:
            validate_employee_input(bad_tenure)
        self.assertIn("cannot exceed possible working years", str(ctx.exception))

    def test_reject_invalid_satisfaction_scores(self):
        for score in [0, 6, -1]:
            bad_score = self.valid_record.copy()
            bad_score["JobSatisfaction"] = score
            with self.assertRaises(ValidationError):
                validate_employee_input(bad_score)

    def test_reject_invalid_overtime(self):
        bad_ot = self.valid_record.copy()
        bad_ot["OverTime"] = "Maybe"
        with self.assertRaises(ValidationError):
            validate_employee_input(bad_ot)


class TestPredictionModule(unittest.TestCase):
    """Tests for model prediction, probabilities, and packaging."""

    @classmethod
    def setUpClass(cls):
        # Generate temporary dataset and train model if not present
        if not MODEL_FILE.exists():
            from src.generate_data import generate_synthetic_data, save_dataset
            from src.train_model import train_and_evaluate_models
            from src.utils import DATA_FILE
            if not DATA_FILE.exists():
                df = generate_synthetic_data(n_samples=200, random_seed=42)
                save_dataset(df, DATA_FILE, overwrite=True)
            train_and_evaluate_models(data_path=DATA_FILE)

    def test_model_loading_and_prediction(self):
        predictor = AttritionPredictor()
        self.assertIsNotNone(predictor.pipeline)

        sample = {
            "Age": 29,
            "Department": "Sales",
            "MonthlyIncome": 35000,
            "YearsAtCompany": 2,
            "JobSatisfaction": 2,
            "WorkLifeBalance": 2,
            "PerformanceRating": 3,
            "OverTime": "Yes",
            "DistanceFromHome": 25.0,
        }

        result = predictor.predict_single(sample)
        self.assertIn(result["predicted_attrition"], ["Yes", "No"])
        self.assertIn(result["prediction_status"], ["Likely to Leave", "Likely to Stay"])

        # Check probability bounds
        p_att = result["attrition_probability_pct"]
        p_stay = result["stay_probability_pct"]
        self.assertGreaterEqual(p_att, 0.0)
        self.assertLessEqual(p_att, 100.0)
        self.assertGreaterEqual(p_stay, 0.0)
        self.assertLessEqual(p_stay, 100.0)
        self.assertAlmostEqual(p_att + p_stay, 100.0, places=1)


if __name__ == "__main__":
    unittest.main()
