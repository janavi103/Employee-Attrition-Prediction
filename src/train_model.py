"""Machine Learning Training, Hyperparameter Tuning, and Evaluation Module.

Trains and rigorously compares Logistic Regression and Random Forest models
using cross-validated hyperparameter tuning, evaluates them on an isolated
stratified test set, logs metrics, generates feature importances, and persists
the champion pipeline with metadata to 'models/best_model.joblib'.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.eda import plot_feature_importance
from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_MAPPING,
    create_preprocessor,
    get_feature_names_from_preprocessor,
    prepare_train_test_data,
)
from src.utils import (
    COMPARISON_FILE,
    DATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    VISUALS_DIR,
)


def build_candidate_pipelines() -> Dict[str, Tuple[Pipeline, Dict[str, list]]]:
    """Define candidate pipelines and their respective hyperparameter search grids.

    Returns:
        Dict: Mapping of model names to (base_pipeline, param_grid).
    """
    candidates = {}

    # Candidate 1: Logistic Regression
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
    }
    candidates["Logistic Regression"] = (lr_pipeline, lr_param_grid)

    # Candidate 2: Random Forest Classifier
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    rf_param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [4, 6, 8, None],
        "classifier__min_samples_split": [2, 5, 10],
    }
    candidates["Random Forest"] = (rf_pipeline, rf_param_grid)

    return candidates


def extract_model_feature_importance(
    pipeline: Pipeline,
) -> Tuple[List[str], List[float]]:
    """Extract feature importances or standardized coefficient magnitudes.

    Args:
        pipeline: Fitted scikit-learn pipeline with 'preprocessor' and 'classifier'.

    Returns:
        Tuple[List[str], List[float]]: Feature names and their normalized weights.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        raw_importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        raw_importances = np.abs(classifier.coef_[0])
    else:
        raw_importances = np.ones(len(feature_names))

    # Normalize to sum to 1.0 for comparability
    norm_importances = (raw_importances / np.sum(raw_importances)).tolist()
    return feature_names, norm_importances


def train_and_evaluate_models(
    data_path: Path = DATA_FILE,
    scoring_metric: str = "f1",
) -> Tuple[pd.DataFrame, Dict[str, Any], Pipeline]:
    """Train candidates with cross-validation, evaluate on test set, and select the best.

    Args:
        data_path: Path to dataset CSV.
        scoring_metric: Metric to optimize in GridSearch (default: 'f1').

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], Pipeline]:
            Comparison DataFrame, best model evaluation metadata, and best fitted pipeline.
    """
    print("\n" + "=" * 70)
    print("      MACHINE LEARNING MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 70)

    # 1. Prepare Stratified Train / Test split (NO data leakage)
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        file_path=data_path,
        test_size=0.20,
        random_seed=42,
    )

    candidates = build_candidate_pipelines()
    results_list = []
    fitted_searches = {}

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (pipeline, param_grid) in candidates.items():
        print(f"\n[TRAINING] Performing GridSearchCV on '{name}'...")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring_metric,
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        fitted_searches[name] = grid_search

        best_estimator = grid_search.best_estimator_
        cv_score = grid_search.best_score_
        print(f"           Best CV {scoring_metric.upper()}: {cv_score:.4f}")
        print(f"           Best Hyperparameters: {grid_search.best_params_}")

        # 2. Evaluate strictly on the unseen test set
        y_pred = best_estimator.predict(X_test)
        y_prob = (
            best_estimator.predict_proba(X_test)[:, 1]
            if hasattr(best_estimator, "predict_proba")
            else None
        )

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")

        print(f"           Test Accuracy : {acc:.4f}")
        print(f"           Test Precision: {prec:.4f}")
        print(f"           Test Recall   : {rec:.4f}")
        print(f"           Test F1-Score : {f1:.4f}")
        print(f"           Test ROC-AUC  : {roc_auc:.4f}")

        results_list.append(
            {
                "Model": name,
                "Best_Params": str(grid_search.best_params_),
                f"CV_{scoring_metric.upper()}": round(cv_score, 4),
                "Test_Accuracy": round(acc, 4),
                "Test_Precision": round(prec, 4),
                "Test_Recall": round(rec, 4),
                "Test_F1_Score": round(f1, 4),
                "Test_ROC_AUC": round(roc_auc, 4),
            }
        )

    # 3. Model Comparison Table
    comparison_df = pd.DataFrame(results_list)
    comparison_df.to_csv(COMPARISON_FILE, index=False)
    print(f"\n[SUCCESS] Model comparison saved to '{COMPARISON_FILE}'.")
    print(comparison_df.to_string(index=False))

    # 4. Selection Rule:
    # We choose the model with the highest Test F1-Score (balancing Precision and Recall for attrition).
    # If tied, we break ties by CV F1-Score.
    best_row = comparison_df.sort_values(
        by=["Test_F1_Score", f"CV_{scoring_metric.upper()}"], ascending=False
    ).iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = fitted_searches[best_model_name].best_estimator_

    print("\n" + "=" * 70)
    print(f"CHAMPION MODEL SELECTED: {best_model_name}")
    print(f"Selection Rationale: Maximizes Test F1-Score ({best_row['Test_F1_Score']:.4f})")
    print("giving the best operational balance between catching flight risks and precision.")
    print("=" * 70)

    # Detailed test metrics for champion model
    y_pred_best = best_pipeline.predict(X_test)
    y_prob_best = best_pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred_best).tolist()
    clf_report = classification_report(y_test, y_pred_best, output_dict=True)

    # Extract feature importances
    feature_names, norm_importances = extract_model_feature_importance(best_pipeline)

    # Plot Plot 4: Feature Importance
    feat_plot_path = plot_feature_importance(
        feature_names=feature_names,
        importances=norm_importances,
        model_name=best_model_name,
        output_dir=VISUALS_DIR,
    )
    print(f"[INFO] Generated feature importance plot at '{feat_plot_path}'.")

    # 5. Save best model artifacts & metadata
    best_metadata = {
        "model_name": best_model_name,
        "selection_rule": "Highest Test F1-Score with Stratified 5-Fold Cross-Validation",
        "best_hyperparameters": fitted_searches[best_model_name].best_params_,
        "test_metrics": {
            "accuracy": float(best_row["Test_Accuracy"]),
            "precision": float(best_row["Test_Precision"]),
            "recall": float(best_row["Test_Recall"]),
            "f1_score": float(best_row["Test_F1_Score"]),
            "roc_auc": float(best_row["Test_ROC_AUC"]),
            "confusion_matrix": cm,
            "classification_report": clf_report,
        },
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": TARGET_MAPPING,
        "feature_names_transformed": feature_names,
        "feature_importances": dict(zip(feature_names, norm_importances)),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(best_metadata, f, indent=4)
    print(f"[SUCCESS] Saved champion metrics to '{METRICS_FILE}'.")

    # Save complete pipeline package using joblib
    model_package = {
        "pipeline": best_pipeline,
        "metadata": best_metadata,
    }
    joblib.dump(model_package, MODEL_FILE)
    print(f"[SUCCESS] Saved complete fitted pipeline package to '{MODEL_FILE}'.")

    return comparison_df, best_metadata, best_pipeline


def main():
    """Command-line entry point for model training."""
    if not DATA_FILE.exists():
        print(f"[ERROR] Dataset '{DATA_FILE}' not found. Please run data generation first:")
        print("        python -m src.generate_data")
        return
    train_and_evaluate_models()


if __name__ == "__main__":
    main()
