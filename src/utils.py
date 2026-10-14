"""Utility functions, path configurations, and input validation for Employee Attrition Prediction.

This module provides common project paths, custom exception classes, and robust
validation routines for individual prediction inputs as well as batch datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

# Define base project directory paths using pathlib
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALS_DIR = PROJECT_ROOT / "visuals"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

DATA_FILE = DATA_DIR / "employee_data.csv"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = REPORTS_DIR / "best_model_metrics.json"
COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"

# Ensure essential output directories exist
for folder in [DATA_DIR, MODELS_DIR, VISUALS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Required features and schema constants
FEATURE_COLUMNS = [
    "Age",
    "Department",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction",
    "WorkLifeBalance",
    "PerformanceRating",
    "OverTime",
    "DistanceFromHome",
]

ALL_DATA_COLUMNS = ["EmployeeID"] + FEATURE_COLUMNS + ["Attrition"]

VALID_DEPARTMENTS = ["Sales", "Engineering", "HR", "Marketing", "Finance"]
VALID_OVERTIME = ["Yes", "No"]
VALID_ATTRITION = ["Yes", "No"]

# Domain boundaries for employee input validation
AGE_MIN, AGE_MAX = 18, 70
INCOME_MIN, INCOME_MAX = 10000, 1000000
RATING_MIN, RATING_MAX = 1, 5
DISTANCE_MIN, DISTANCE_MAX = 0, 150
TENURE_MIN = 0


class ValidationError(ValueError):
    """Custom exception raised when user inputs or schema checks fail validation."""
    pass


class DataError(Exception):
    """Custom exception raised when dataset loading or integrity checks fail."""
    pass


class ModelError(Exception):
    """Custom exception raised when model loading or artifact files are missing or invalid."""
    pass


def validate_employee_input(record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single employee record for real-time prediction.

    Args:
        record: Dictionary containing the required feature values.

    Returns:
        Dict[str, Any]: Sanitized and validated dictionary with typed values.

    Raises:
        ValidationError: If any required feature is missing, out-of-range, or has an invalid type.
    """
    if not isinstance(record, dict):
        raise ValidationError(f"Employee record must be a dictionary, got {type(record).__name__}.")

    # 1. Check for missing keys
    missing_fields = [col for col in FEATURE_COLUMNS if col not in record or record[col] is None]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}.")

    validated = {}

    # 2. Validate Age
    try:
        age = int(record["Age"])
    except (ValueError, TypeError):
        raise ValidationError(f"Age must be an integer, got '{record['Age']}'.")
    if not (AGE_MIN <= age <= AGE_MAX):
        raise ValidationError(
            f"Age must be between {AGE_MIN} and {AGE_MAX} years. Received: {age}."
        )
    validated["Age"] = age

    # 3. Validate Department
    dept = str(record["Department"]).strip()
    if dept not in VALID_DEPARTMENTS:
        raise ValidationError(
            f"Department must be one of {VALID_DEPARTMENTS}. Received: '{dept}'."
        )
    validated["Department"] = dept

    # 4. Validate MonthlyIncome
    try:
        income = float(record["MonthlyIncome"])
    except (ValueError, TypeError):
        raise ValidationError(f"MonthlyIncome must be numeric, got '{record['MonthlyIncome']}'.")
    if not (INCOME_MIN <= income <= INCOME_MAX):
        raise ValidationError(
            f"MonthlyIncome must be between INR {INCOME_MIN:,} and INR {INCOME_MAX:,}. Received: INR {income:,.2f}."
        )
    validated["MonthlyIncome"] = income

    # 5. Validate YearsAtCompany (Tenure)
    try:
        tenure = int(record["YearsAtCompany"])
    except (ValueError, TypeError):
        raise ValidationError(f"YearsAtCompany must be an integer, got '{record['YearsAtCompany']}'.")
    if tenure < TENURE_MIN:
        raise ValidationError(f"YearsAtCompany cannot be negative. Received: {tenure}.")
    max_tenure_for_age = age - AGE_MIN
    if tenure > max_tenure_for_age:
        raise ValidationError(
            f"YearsAtCompany ({tenure}) cannot exceed possible working years for age {age} "
            f"(maximum possible tenure: {max_tenure_for_age} years)."
        )
    validated["YearsAtCompany"] = tenure

    # 6. Validate JobSatisfaction, WorkLifeBalance, PerformanceRating (1 to 5)
    score_fields = ["JobSatisfaction", "WorkLifeBalance", "PerformanceRating"]
    for sf in score_fields:
        try:
            val = int(record[sf])
        except (ValueError, TypeError):
            raise ValidationError(f"{sf} must be an integer between 1 and 5, got '{record[sf]}'.")
        if not (RATING_MIN <= val <= RATING_MAX):
            raise ValidationError(
                f"{sf} score must be between {RATING_MIN} and {RATING_MAX}. Received: {val}."
            )
        validated[sf] = val

    # 7. Validate OverTime
    ot = str(record["OverTime"]).strip()
    if ot not in VALID_OVERTIME:
        raise ValidationError(
            f"OverTime must be one of {VALID_OVERTIME}. Received: '{ot}'."
        )
    validated["OverTime"] = ot

    # 8. Validate DistanceFromHome
    try:
        dist = float(record["DistanceFromHome"])
    except (ValueError, TypeError):
        raise ValidationError(f"DistanceFromHome must be numeric, got '{record['DistanceFromHome']}'.")
    if not (DISTANCE_MIN <= dist <= DISTANCE_MAX):
        raise ValidationError(
            f"DistanceFromHome must be between {DISTANCE_MIN} and {DISTANCE_MAX} km. Received: {dist}."
        )
    validated["DistanceFromHome"] = dist

    return validated


def validate_dataset_schema(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, List[str]]:
    """Validate that a Pandas DataFrame contains expected columns and valid data structure.

    Args:
        df: Pandas DataFrame to check.
        require_target: Whether the 'Attrition' column is mandatory.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_error_messages)
    """
    errors = []
    if df is None or not isinstance(df, pd.DataFrame):
        return False, ["Input is not a valid Pandas DataFrame."]

    if df.empty:
        return False, ["Dataset is empty (0 rows)."]

    expected_cols = list(FEATURE_COLUMNS)
    if require_target:
        expected_cols.append("Attrition")

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if "Department" in df.columns:
        invalid_depts = df[~df["Department"].isin(VALID_DEPARTMENTS)]["Department"].dropna().unique()
        if len(invalid_depts) > 0:
            errors.append(f"Invalid department categories found: {list(invalid_depts)}")

    if "OverTime" in df.columns:
        invalid_ot = df[~df["OverTime"].isin(VALID_OVERTIME)]["OverTime"].dropna().unique()
        if len(invalid_ot) > 0:
            errors.append(f"Invalid OverTime values found: {list(invalid_ot)}")

    if require_target and "Attrition" in df.columns:
        invalid_att = df[~df["Attrition"].isin(VALID_ATTRITION)]["Attrition"].dropna().unique()
        if len(invalid_att) > 0:
            errors.append(f"Invalid Attrition target values found: {list(invalid_att)}")

    return len(errors) == 0, errors
