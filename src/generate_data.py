"""Synthetic Employee Dataset Generator for Attrition Prediction.

Generates a realistic, reproducible synthetic dataset of approximately 1,000 employee
records saved to 'data/employee_data.csv'. Includes realistic correlations, imperfect
attrition probabilities, and an authentic class imbalance ratio.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils import (
    DATA_FILE,
    FEATURE_COLUMNS,
    VALID_DEPARTMENTS,
)


def generate_synthetic_data(
    n_samples: int = 1000,
    random_seed: int = 42,
    target_attrition_rate: float = 0.17,
) -> pd.DataFrame:
    """Generate a synthetic employee dataset with realistic organizational patterns.

    Args:
        n_samples: Number of employee records to generate (default: 1000).
        random_seed: Random seed for exact reproducibility (default: 42).
        target_attrition_rate: Approximate expected attrition rate (default: ~17%).

    Returns:
        pd.DataFrame: Generated dataset containing employee demographics and attrition target.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Employee Identifiers
    employee_ids = [f"EMP{i:04d}" for i in range(1, n_samples + 1)]

    # 2. Demographics & Tenure
    # Age distribution between 21 and 60 with mean ~36
    ages = rng.integers(low=21, high=61, size=n_samples)

    # YearsAtCompany cannot exceed age - 18, bounded between 0 and 35
    max_tenure_possible = np.maximum(0, ages - 18)
    # Most employees have between 1 and 15 years tenure
    raw_tenure = rng.geometric(p=0.15, size=n_samples) - 1
    years_at_company = np.minimum(raw_tenure, max_tenure_possible)

    # 3. Department
    dept_weights = [0.30, 0.35, 0.10, 0.15, 0.10]  # Sales, Engineering, HR, Marketing, Finance
    departments = rng.choice(VALID_DEPARTMENTS, size=n_samples, p=dept_weights)

    # 4. Monthly Income in INR
    # Base salary varies by department and scales with age and tenure
    dept_base_salary = {
        "Sales": 48000,
        "Engineering": 65000,
        "HR": 42000,
        "Marketing": 46000,
        "Finance": 58000,
    }

    monthly_incomes = []
    for i in range(n_samples):
        base = dept_base_salary[departments[i]]
        # Experience multiplier
        exp_boost = (years_at_company[i] * 2800) + ((ages[i] - 21) * 1100)
        # Random variation
        variation = rng.normal(loc=0, scale=8000)
        income = max(18000.0, round(base + exp_boost + variation, -2))
        monthly_incomes.append(float(income))
    monthly_incomes = np.array(monthly_incomes)

    # 5. OverTime (approx 28% work overtime)
    overtime_flags = rng.choice(["Yes", "No"], size=n_samples, p=[0.28, 0.72])

    # 6. DistanceFromHome (1 to 45 km)
    distances = rng.integers(low=1, high=46, size=n_samples)

    # 7. Job Satisfaction, Work-Life Balance, Performance Rating (1 to 5)
    # Job satisfaction slightly lower for overtime workers
    job_satisfaction = []
    work_life_balance = []
    performance_rating = []

    for i in range(n_samples):
        is_ot = overtime_flags[i] == "Yes"

        # Work-Life Balance (1-5, lower if OverTime)
        wlb_p = [0.15, 0.35, 0.30, 0.15, 0.05] if is_ot else [0.03, 0.12, 0.35, 0.35, 0.15]
        wlb = rng.choice([1, 2, 3, 4, 5], p=wlb_p)
        work_life_balance.append(int(wlb))

        # Job Satisfaction (1-5)
        js_p = [0.12, 0.23, 0.30, 0.25, 0.10] if is_ot else [0.06, 0.16, 0.32, 0.32, 0.14]
        js = rng.choice([1, 2, 3, 4, 5], p=js_p)
        job_satisfaction.append(int(js))

        # Performance Rating (1-5, centered at 3-4)
        pr_p = [0.03, 0.12, 0.50, 0.28, 0.07]
        pr = rng.choice([1, 2, 3, 4, 5], p=pr_p)
        performance_rating.append(int(pr))

    job_satisfaction = np.array(job_satisfaction)
    work_life_balance = np.array(work_life_balance)
    performance_rating = np.array(performance_rating)

    # 8. Realistic imperfect Attrition probability generation
    # Normalize components for log-odds calculation
    norm_income = (monthly_incomes - np.mean(monthly_incomes)) / np.std(monthly_incomes)
    norm_distance = (distances - np.mean(distances)) / np.std(distances)
    norm_tenure = (years_at_company - np.mean(years_at_company)) / np.std(years_at_company)
    norm_age = (ages - np.mean(ages)) / np.std(ages)

    # Base log-odds calibrated for ~17% attrition rate
    # Factors increasing attrition:
    # - OverTime == 'Yes' (+1.15)
    # - Low JobSatisfaction (-0.45 per point below 3)
    # - Low WorkLifeBalance (-0.35 per point below 3)
    # - High DistanceFromHome (+0.25)
    # - Low MonthlyIncome (-0.30)
    # - Lower Age / Lower Tenure (-0.25)
    # - Noise (+ random normal with std=0.85)

    ot_effect = np.where(overtime_flags == "Yes", 1.25, -0.40)
    js_effect = -0.48 * (job_satisfaction - 3.0)
    wlb_effect = -0.38 * (work_life_balance - 3.0)
    income_effect = -0.32 * norm_income
    distance_effect = 0.26 * norm_distance
    tenure_effect = -0.22 * norm_tenure
    age_effect = -0.20 * norm_age
    noise = rng.normal(loc=0.0, scale=0.85, size=n_samples)

    # Department-specific subtle variance
    dept_bias = {
        "Sales": 0.25,
        "Engineering": 0.05,
        "HR": 0.15,
        "Marketing": 0.10,
        "Finance": -0.20,
    }
    dept_effect = np.array([dept_bias[d] for d in departments])

    # Intercept to achieve desired baseline
    intercept = -1.95

    log_odds = (
        intercept
        + ot_effect
        + js_effect
        + wlb_effect
        + income_effect
        + distance_effect
        + tenure_effect
        + age_effect
        + dept_effect
        + noise
    )

    # Sigmoid function for probability
    prob_attrition = 1.0 / (1.0 + np.exp(-log_odds))

    # Binary outcome based on Bernoulli trial using probabilities
    attrition_binary = rng.binomial(n=1, p=prob_attrition)
    attrition = np.where(attrition_binary == 1, "Yes", "No")

    # Construct DataFrame
    df = pd.DataFrame(
        {
            "EmployeeID": employee_ids,
            "Age": ages,
            "Department": departments,
            "MonthlyIncome": monthly_incomes,
            "YearsAtCompany": years_at_company,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "PerformanceRating": performance_rating,
            "OverTime": overtime_flags,
            "DistanceFromHome": distances,
            "Attrition": attrition,
        }
    )

    return df


def save_dataset(df: pd.DataFrame, output_path: Path = DATA_FILE, overwrite: bool = False) -> Path:
    """Save the dataset to CSV file if it does not already exist or overwrite is specified.

    Args:
        df: DataFrame to save.
        output_path: Target CSV file destination.
        overwrite: Whether to overwrite an existing file.

    Returns:
        Path: Path to saved file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"[INFO] Dataset already exists at '{output_path}'. Use --overwrite to regenerate.")
        return output_path

    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Synthetic dataset generated and saved to '{output_path}'.")
    print(f"          Total records: {len(df):,}")
    attrition_count = (df['Attrition'] == 'Yes').sum()
    attrition_pct = (attrition_count / len(df)) * 100
    print(f"          Attrition distribution: {attrition_count} Left ({attrition_pct:.1f}%), "
          f"{len(df) - attrition_count} Stayed ({100 - attrition_pct:.1f}%)")
    return output_path


def main():
    """Command-line entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic employee dataset for attrition prediction."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of synthetic records to generate (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite of existing dataset CSV",
    )
    args = parser.parse_args()

    if DATA_FILE.exists() and not args.overwrite:
        print(f"[INFO] '{DATA_FILE}' already exists. Skipping generation. Use --overwrite to replace.")
        return

    df = generate_synthetic_data(n_samples=args.samples, random_seed=args.seed)
    save_dataset(df, output_path=DATA_FILE, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
