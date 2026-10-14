"""Data Preprocessing Pipeline for Employee Attrition Prediction.

This module provides data loading, data cleaning, integrity validation,
stratified train-test splitting, and Scikit-learn Pipeline construction
with ColumnTransformer (handling numeric scaling, categorical encoding,
and missing value imputation).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    DATA_FILE,
    FEATURE_COLUMNS,
    DataError,
    validate_dataset_schema,
)

# Feature group definitions
NUMERICAL_FEATURES: List[str] = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction",
    "WorkLifeBalance",
    "PerformanceRating",
    "DistanceFromHome",
]

CATEGORICAL_FEATURES: List[str] = [
    "Department",
    "OverTime",
]

TARGET_COLUMN: str = "Attrition"
TARGET_MAPPING: Dict[str, int] = {"No": 0, "Yes": 1}
REVERSE_TARGET_MAPPING: Dict[int, str] = {0: "No", 1: "Yes"}


def load_dataset(file_path: Union[str, Path] = DATA_FILE) -> pd.DataFrame:
    """Load employee dataset from CSV file.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        DataError: If file is missing or unreadable.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataError(f"Dataset file not found at: '{path}'. Please run data generation first.")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataError(f"Failed to read dataset from '{path}': {str(e)}")

    return df


def clean_and_validate_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    require_target: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean dataset, remove duplicates, report missing values, and validate schema.

    Args:
        df: Input DataFrame.
        drop_duplicates: Whether to drop duplicate rows based on features.
        require_target: Whether target column is expected.

    Returns:
        Tuple[pd.DataFrame, Dict[str, int]]: Cleaned DataFrame and missing values summary.

    Raises:
        DataError: If dataset fails schema validation or is empty.
    """
    if df is None or df.empty:
        raise DataError("Cannot clean empty DataFrame.")

    # Validate schema
    is_valid, errors = validate_dataset_schema(df, require_target=require_target)
    if not is_valid:
        raise DataError(f"Dataset validation failed:\n - " + "\n - ".join(errors))

    initial_count = len(df)
    df_clean = df.copy()

    # Track missing values
    missing_summary = df_clean.isnull().sum().to_dict()

    # Remove duplicates (ignoring EmployeeID if present)
    dedup_cols = [c for c in FEATURE_COLUMNS if c in df_clean.columns]
    if drop_duplicates and dedup_cols:
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
        dropped_dups = before_dedup - len(df_clean)
        if dropped_dups > 0:
            print(f"[INFO] Removed {dropped_dups} duplicate records.")

    return df_clean, missing_summary


def create_preprocessor() -> ColumnTransformer:
    """Create a Scikit-learn ColumnTransformer for preprocessing features.

    Numerical pipeline:
      - Median imputation for missing values
      - Standard scaling

    Categorical pipeline:
      - Most-frequent imputation for missing values
      - OneHotEncoder with handle_unknown='ignore'

    Returns:
        ColumnTransformer: Preprocessor ready to be incorporated into a Pipeline.
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Automatically drops non-feature columns like EmployeeID
    )

    return preprocessor


def prepare_train_test_data(
    file_path: Union[str, Path] = DATA_FILE,
    test_size: float = 0.20,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, validate, clean, and split the dataset into stratified train and test partitions.

    Important: Does NOT fit transformations on test data to prevent data leakage.

    Args:
        file_path: Path to employee dataset CSV.
        test_size: Ratio of test partition (default: 0.20).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    df = load_dataset(file_path)
    df_clean, missing_summary = clean_and_validate_data(df, drop_duplicates=True, require_target=True)

    # Separate features and target
    X = df_clean[FEATURE_COLUMNS].copy()
    y_raw = df_clean[TARGET_COLUMN].copy()

    # Map target variable to binary (1 = Yes / Left, 0 = No / Stayed)
    y = y_raw.map(TARGET_MAPPING).astype(int)

    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    print(f"[INFO] Dataset split successfully:")
    print(f"       Total records: {len(df_clean)}")
    print(f"       Training set: {len(X_train)} samples ({y_train.sum()} Attrition = Yes)")
    print(f"       Testing set:  {len(X_test)} samples ({y_test.sum()} Attrition = Yes)")

    return X_train, X_test, y_train, y_test


def get_feature_names_from_preprocessor(
    fitted_preprocessor: ColumnTransformer,
) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer.

    Args:
        fitted_preprocessor: Preprocessor fitted on training data.

    Returns:
        List[str]: List of transformed feature names.
    """
    try:
        # Scikit-learn >= 1.0 feature name extraction
        feature_names = fitted_preprocessor.get_feature_names_out()
        # Clean prefix names like 'num__' and 'cat__' for cleaner display
        clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
        return clean_names
    except Exception:
        # Fallback if get_feature_names_out fails
        cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        return list(NUMERICAL_FEATURES) + list(cat_features)
