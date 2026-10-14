"""Exploratory Data Analysis (EDA) Module for Employee Attrition.

Generates comprehensive statistical summaries and publishes key publication-quality
visualizations saved to the 'visuals/' directory.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocessing import clean_and_validate_data, load_dataset
from src.utils import DATA_FILE, VISUALS_DIR

# Set global aesthetic style for charts
sns.set_theme(style="whitegrid", font="sans-serif")
PALETTE_PRIMARY = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
PALETTE_BINARY = ["#1f77b4", "#ff7f0e"]  # No vs Yes


def plot_attrition_by_department(df: pd.DataFrame, output_dir: Path = VISUALS_DIR) -> Path:
    """Plot Plot 1: Attrition rate (%) across departments."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    dept_stats = (
        df.groupby("Department")["Attrition"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="AttritionRate")
        .sort_values(by="AttritionRate", ascending=False)
    )

    bars = ax.bar(dept_stats["Department"], dept_stats["AttritionRate"], color="#2b5c8f", width=0.55)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Employee Attrition Rate by Department", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Department", fontsize=11, labelpad=10)
    ax.set_ylabel("Attrition Rate (%)", fontsize=11, labelpad=10)
    ax.set_ylim(0, max(dept_stats["AttritionRate"]) * 1.25)
    plt.tight_layout()

    out_file = output_dir / "attrition_by_department.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_income_vs_attrition(df: pd.DataFrame, output_dir: Path = VISUALS_DIR) -> Path:
    """Plot Plot 2: Monthly Income distribution by Attrition status."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    sns.boxplot(
        data=df,
        x="Attrition",
        y="MonthlyIncome",
        hue="Attrition",
        legend=False,
        palette={"No": "#2ca02c", "Yes": "#d62728"},
        width=0.45,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "yellow", "markeredgecolor": "black"},
    )

    ax.set_title("Monthly Income Distribution by Attrition Status", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Attrition Status (Left Organization?)", fontsize=11, labelpad=10)
    ax.set_ylabel("Monthly Income (INR)", fontsize=11, labelpad=10)
    ax.yaxis.set_major_formatter("INR {x:,.0f}")
    
    # Annotate median values
    medians = df.groupby("Attrition")["MonthlyIncome"].median()
    for i, (status, med_val) in enumerate(medians.items()):
        ax.text(
            i,
            med_val,
            f"  Median: INR {med_val:,.0f}",
            color="black",
            weight="semibold",
            va="center",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    out_file = output_dir / "income_vs_attrition.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path = VISUALS_DIR) -> Path:
    """Plot Plot 3: Correlation heatmap of numerical features and binary Attrition."""
    fig, ax = plt.subplots(figsize=(9, 7))

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if "EmployeeID" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["EmployeeID"])

    # Include binary attrition for correlation analysis
    if "Attrition" in df.columns:
        numeric_df["Attrition_Binary"] = (df["Attrition"] == "Yes").astype(int)

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (Association)"},
        ax=ax,
    )

    ax.set_title(
        "Correlation Matrix of Employee Metrics (Association, Not Causation)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    out_file = output_dir / "correlation_heatmap.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_attrition_by_overtime(df: pd.DataFrame, output_dir: Path = VISUALS_DIR) -> Path:
    """Plot Plot 5: Attrition rate by OverTime status."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ot_stats = (
        df.groupby("OverTime")["Attrition"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="AttritionRate")
    )

    bars = ax.bar(
        ot_stats["OverTime"],
        ot_stats["AttritionRate"],
        color=["#4575b4", "#d73027"],
        width=0.45,
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_title("Attrition Rate by Overtime Requirement", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Works Overtime?", fontsize=11, labelpad=10)
    ax.set_ylabel("Attrition Rate (%)", fontsize=11, labelpad=10)
    ax.set_ylim(0, max(ot_stats["AttritionRate"]) * 1.3)
    plt.tight_layout()

    out_file = output_dir / "attrition_by_overtime.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_attrition_by_satisfaction(df: pd.DataFrame, output_dir: Path = VISUALS_DIR) -> Path:
    """Plot Plot 6: Attrition rate across Job Satisfaction ratings (1 to 5)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sat_stats = (
        df.groupby("JobSatisfaction")["Attrition"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="AttritionRate")
    )

    colors = sns.color_palette("RdYlGn_r", n_colors=len(sat_stats))
    bars = ax.bar(sat_stats["JobSatisfaction"], sat_stats["AttritionRate"], color=colors, width=0.55)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Attrition Rate by Job Satisfaction Score", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Job Satisfaction (1 = Very Dissatisfied, 5 = Highly Satisfied)", fontsize=11, labelpad=10)
    ax.set_ylabel("Attrition Rate (%)", fontsize=11, labelpad=10)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, max(sat_stats["AttritionRate"]) * 1.3)
    plt.tight_layout()

    out_file = output_dir / "attrition_by_satisfaction.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def plot_feature_importance(
    feature_names: list,
    importances: list,
    model_name: str,
    output_dir: Path = VISUALS_DIR,
) -> Path:
    """Plot Plot 4: Feature Importance or Coefficient Magnitude for the best model."""
    fig, ax = plt.subplots(figsize=(9, 6))

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=True)

    y_pos = np.arange(len(importance_df))
    bars = ax.barh(y_pos, importance_df["Importance"], color="#3182bd", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df["Feature"], fontsize=10)
    ax.set_xlabel("Relative Importance / Weight Magnitude", fontsize=11, labelpad=10)
    ax.set_title(f"Model Feature Importance ({model_name})", fontsize=14, fontweight="bold", pad=15)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.3f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    out_file = output_dir / "feature_importance.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def run_eda(file_path: Path = DATA_FILE, output_dir: Path = VISUALS_DIR) -> None:
    """Execute complete Exploratory Data Analysis and generate all visual plots.

    Args:
        file_path: Path to employee dataset CSV.
        output_dir: Target directory for generated PNG figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(file_path)
    df_clean, _ = clean_and_validate_data(df, drop_duplicates=True, require_target=True)

    total_records = len(df_clean)
    attrition_count = (df_clean["Attrition"] == "Yes").sum()
    attrition_rate = (attrition_count / total_records) * 100

    print("=" * 70)
    print("        EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 70)
    print(f"Total Employees Analyzed: {total_records:,}")
    print(f"Attrition Count: {attrition_count} left ({attrition_rate:.2f}%)")
    print(f"Retention Count: {total_records - attrition_count} stayed ({100 - attrition_rate:.2f}%)")
    print("-" * 70)

    # Department breakdown
    print("\n1. Attrition Rate by Department:")
    dept_rates = df_clean.groupby("Department")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    for dept, rate in dept_rates.items():
        print(f"   - {dept:12s}: {rate:5.1f}%")

    # Overtime breakdown
    print("\n2. Attrition Rate by Overtime:")
    ot_rates = df_clean.groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    for ot, rate in ot_rates.items():
        print(f"   - Overtime {ot:3s}: {rate:5.1f}%")

    # Satisfaction breakdown
    print("\n3. Attrition Rate by Job Satisfaction Score (1-5):")
    sat_rates = df_clean.groupby("JobSatisfaction")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
    for score, rate in sat_rates.items():
        print(f"   - Score {score}: {rate:5.1f}%")

    # Income stats
    print("\n4. Monthly Income Statistics:")
    inc_left = df_clean[df_clean["Attrition"] == "Yes"]["MonthlyIncome"]
    inc_stay = df_clean[df_clean["Attrition"] == "No"]["MonthlyIncome"]
    print(f"   - Employees who Left   : Mean = INR {inc_left.mean():,.0f}, Median = INR {inc_left.median():,.0f}")
    print(f"   - Employees who Stayed : Mean = INR {inc_stay.mean():,.0f}, Median = INR {inc_stay.median():,.0f}")

    # Demographic observations
    print("\n5. Tenure, Age, and Commute Distance Observations:")
    print(f"   - Average Age (Left vs Stayed): {inc_left.index.size} left (Mean Age {df_clean[df_clean['Attrition']=='Yes']['Age'].mean():.1f}) vs {df_clean[df_clean['Attrition']=='No']['Age'].mean():.1f} stayed")
    print(f"   - Average Distance from Home : Left {df_clean[df_clean['Attrition']=='Yes']['DistanceFromHome'].mean():.1f} km vs Stayed {df_clean[df_clean['Attrition']=='No']['DistanceFromHome'].mean():.1f} km")
    print(f"   - Average Years at Company   : Left {df_clean[df_clean['Attrition']=='Yes']['YearsAtCompany'].mean():.1f} yrs vs Stayed {df_clean[df_clean['Attrition']=='No']['YearsAtCompany'].mean():.1f} yrs")
    print("-" * 70)
    print("Note: Reported correlations and rate differences represent observational associations,")
    print("not direct causal relationships. External factors must be considered.")
    print("=" * 70)

    # Save charts
    print("\n[INFO] Saving visualization charts to 'visuals/'...")
    p1 = plot_attrition_by_department(df_clean, output_dir)
    print(f" - Saved: {p1.name}")
    p2 = plot_income_vs_attrition(df_clean, output_dir)
    print(f" - Saved: {p2.name}")
    p3 = plot_correlation_heatmap(df_clean, output_dir)
    print(f" - Saved: {p3.name}")
    p5 = plot_attrition_by_overtime(df_clean, output_dir)
    print(f" - Saved: {p5.name}")
    p6 = plot_attrition_by_satisfaction(df_clean, output_dir)
    print(f" - Saved: {p6.name}")
    print("[SUCCESS] Core EDA visualizations generated successfully.\n")


def main():
    """Command-line entry point for EDA module."""
    if not DATA_FILE.exists():
        print(f"[ERROR] Data file '{DATA_FILE}' does not exist.")
        print("Please generate data first by executing: python -m src.generate_data")
        return
    run_eda(DATA_FILE, VISUALS_DIR)


if __name__ == "__main__":
    main()
