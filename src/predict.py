"""Inference and Prediction Module for Employee Attrition.

Provides reusable prediction routines using the fitted pipeline package,
rigorous schema validation, probability calculations, and a convenient CLI mode.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.utils import (
    FEATURE_COLUMNS,
    MODEL_FILE,
    ModelError,
    ValidationError,
    validate_employee_input,
)


class AttritionPredictor:
    """Predictor service encapsulating loaded pipeline and inference logic."""

    def __init__(self, model_path: Union[str, Path] = MODEL_FILE):
        self.model_path = Path(model_path)
        self.package: Optional[Dict[str, Any]] = None
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.load_model()

    def load_model(self) -> None:
        """Load the serialized model package from disk.

        Raises:
            ModelError: If model file is missing or corrupted.
        """
        if not self.model_path.exists():
            raise ModelError(
                f"Trained model artifact not found at '{self.model_path}'. "
                "Please train the model first by executing: python -m src.train_model"
            )

        try:
            loaded = joblib.load(self.model_path)
            if isinstance(loaded, dict) and "pipeline" in loaded:
                self.package = loaded
                self.pipeline = loaded["pipeline"]
                self.metadata = loaded.get("metadata", {})
            else:
                # Direct pipeline fallback
                self.pipeline = loaded
                self.metadata = {}
        except Exception as e:
            raise ModelError(f"Error loading model from '{self.model_path}': {str(e)}")

    def predict_single(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input and predict attrition for an individual employee record.

        Args:
            employee_data: Raw input dictionary of employee attributes.

        Returns:
            Dict[str, Any]: Prediction result with probabilities and risk assessment.

        Raises:
            ValidationError: If inputs are invalid or missing.
            ModelError: If model is not loaded properly.
        """
        if self.pipeline is None:
            raise ModelError("Model pipeline is not initialized.")

        # 1. Validate employee input fields, types, and domain ranges
        validated_data = validate_employee_input(employee_data)

        # 2. Convert to DataFrame matching feature column order
        df_input = pd.DataFrame([validated_data])[FEATURE_COLUMNS]

        # 3. Predict class and probability
        pred_class_numeric = self.pipeline.predict(df_input)[0]

        # Determine class indices robustly
        classifier = self.pipeline.named_steps.get("classifier")
        if hasattr(classifier, "classes_"):
            classes = list(classifier.classes_)
            pos_index = classes.index(1) if 1 in classes else 1
            neg_index = classes.index(0) if 0 in classes else 0
        else:
            pos_index = 1
            neg_index = 0

        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(df_input)[0]
            prob_attrition = float(probabilities[pos_index])
            prob_stay = float(probabilities[neg_index])
        else:
            # Fallback if probability not supported
            prob_attrition = 1.0 if pred_class_numeric == 1 else 0.0
            prob_stay = 1.0 - prob_attrition

        predicted_label = "Yes" if pred_class_numeric == 1 else "No"
        status_text = "Likely to Leave" if predicted_label == "Yes" else "Likely to Stay"

        return {
            "predicted_attrition": predicted_label,
            "prediction_status": status_text,
            "attrition_probability_pct": round(prob_attrition * 100, 2),
            "stay_probability_pct": round(prob_stay * 100, 2),
            "model_name": self.metadata.get("model_name", "Trained Classifier"),
            "disclaimer": (
                "Notice: This prediction is a probabilistic model estimate based on historical patterns. "
                "It does not guarantee individual employee actions and must not be used as the sole basis "
                "for HR personnel or employment decisions."
            ),
        }


def predict_employee_attrition(employee_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to run a single prediction using default model file."""
    predictor = AttritionPredictor()
    return predictor.predict_single(employee_data)


def interactive_cli():
    """Run an interactive CLI session for attrition prediction."""
    print("=" * 65)
    print("       EMPLOYEE ATTRITION PREDICTION - INTERACTIVE CLI")
    print("=" * 65)

    try:
        predictor = AttritionPredictor()
    except ModelError as e:
        print(f"[ERROR] {e}")
        return

    print("Please choose an option:")
    print(" 1. Run inference on a sample employee record")
    print(" 2. Enter employee details manually")
    choice = input("Enter choice (1 or 2, default: 1): ").strip()

    if choice == "2":
        try:
            print("\nEnter employee attributes:")
            age = int(input("Age (18-70): ").strip())
            dept = input("Department (Sales/Engineering/HR/Marketing/Finance): ").strip()
            income = float(input("Monthly Income in INR: ").strip())
            tenure = int(input(f"Years at Company (0-{age-18}): ").strip())
            js = int(input("Job Satisfaction (1-5): ").strip())
            wlb = int(input("Work-Life Balance (1-5): ").strip())
            perf = int(input("Performance Rating (1-5): ").strip())
            ot = input("OverTime (Yes/No): ").strip()
            dist = float(input("Distance From Home in km (0-150): ").strip())

            record = {
                "Age": age,
                "Department": dept,
                "MonthlyIncome": income,
                "YearsAtCompany": tenure,
                "JobSatisfaction": js,
                "WorkLifeBalance": wlb,
                "PerformanceRating": perf,
                "OverTime": ot,
                "DistanceFromHome": dist,
            }
        except Exception as e:
            print(f"[INPUT ERROR] Invalid format: {e}")
            return
    else:
        # Sample employee with higher flight risk profile
        print("\nUsing sample at-risk employee profile:")
        record = {
            "Age": 28,
            "Department": "Sales",
            "MonthlyIncome": 38000,
            "YearsAtCompany": 2,
            "JobSatisfaction": 2,
            "WorkLifeBalance": 2,
            "PerformanceRating": 3,
            "OverTime": "Yes",
            "DistanceFromHome": 28.0,
        }
        for k, v in record.items():
            print(f"  {k:20s}: {v}")

    try:
        result = predictor.predict_single(record)
        print("\n" + "-" * 65)
        print("PREDICTION RESULT:")
        print(f"  Status                  : {result['prediction_status']}")
        print(f"  Predicted Attrition     : {result['predicted_attrition']}")
        print(f"  Probability of Leaving  : {result['attrition_probability_pct']}%")
        print(f"  Probability of Staying  : {result['stay_probability_pct']}%")
        print(f"  Model Employed          : {result['model_name']}")
        print("-" * 65)
        print(f"{result['disclaimer']}\n")
    except ValidationError as ve:
        print(f"\n[VALIDATION FAILED] {ve}")


def main():
    """Command-line entry point."""
    interactive_cli()


if __name__ == "__main__":
    main()
