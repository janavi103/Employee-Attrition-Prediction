# 👥 Employee Attrition Prediction System

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Decision-Support System developed for the **Infobharat Interns** internship task. This project analyzes organizational HR data—including compensation, job satisfaction, work-life balance, overtime workloads, and employment tenure—to predict employee flight risk, explain underlying organizational patterns, and provide an interactive web dashboard for real-time risk assessment.

---

## 📌 Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Technology Stack](#3-technology-stack)
- [4. Dataset Description](#4-dataset-description)
- [5. Project Directory Structure](#5-project-directory-structure)
- [6. Installation & Setup (Windows / VS Code)](#6-installation--setup-windows--vs-code)
- [7. How to Run the Application](#7-how-to-run-the-application)
- [8. Exploratory Data Analysis (EDA) Insights](#8-exploratory-data-analysis-eda-insights)
- [9. Model Benchmarking & Performance Comparison](#9-model-benchmarking--performance-comparison)
- [10. Model Selection Rationale](#10-model-selection-rationale)
- [11. Governance, Ethics & Limitations](#11-governance-ethics--limitations)
- [12. Future Enhancements](#12-future-enhancements)
- [13. GitHub Submission Instructions](#13-github-submission-instructions)
- [14. Author Information](#14-author-information)

---

## 1. Project Overview

Employee attrition (unplanned turnover) poses substantial financial and operational challenges to organizations, leading to recruitment costs, project delays, and loss of institutional knowledge. 

The **Employee Attrition Prediction System** transitions HR management from reactive exit interviews to proactive, data-informed retention interventions. By combining:
- **Reproducible Data Engineering:** Generating realistic organizational HR datasets with authentic class imbalance (~17% attrition rate).
- **Leakage-Free Preprocessing:** Building Scikit-learn Pipelines with `ColumnTransformer` fitted strictly on training splits.
- **Cross-Validated Machine Learning:** Comparing **Logistic Regression** and **Random Forest** classifiers tuned via 5-fold cross-validation.
- **Interactive Web Interface:** A polished **Streamlit** dashboard enabling HR personnel to evaluate employee flight risk in real time.

> **Important Notice:** The dataset used in this repository is synthetic demonstration data generated for academic and demonstration purposes. It does not contain confidential personal information or real employee records.

---

## 2. Key Features

- **Automated Data Generation (`src/generate_data.py`):** Generates 1,000 reproducible employee records with realistic relationships and domain bounds.
- **Robust Preprocessing (`src/preprocessing.py`):** Median/mode imputers, `StandardScaler`, and `OneHotEncoder` encapsulated in a Scikit-learn Pipeline with zero data leakage.
- **Exploratory Data Analysis (`src/eda.py`):** Produces 6 high-resolution charts examining department attrition, income box plots, satisfaction trends, overtime impact, correlation heatmaps, and feature importances.
- **Hyperparameter Optimization (`src/train_model.py`):** 5-Fold Stratified `GridSearchCV` optimizing for Test F1-Score to balance Recall and Precision under class imbalance.
- **Real-Time Inference (`src/predict.py`):** Predicts attrition status (`Likely to Leave` vs `Likely to Stay`), exact confidence probabilities, and full input domain validation.
- **Interactive Dashboard (`app.py`):** Responsive multi-tab Streamlit web application with KPI metrics, exploratory visualizations, model comparison tables, and prediction forms.
- **Automated Test Suite (`tests/test_project.py`):** Automated tests verifying dataset schema, pipeline transformations, edge-case rejection, and inference validity.

---

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.11** | Core programming language for all ML, processing, and visualization logic |
| **Visual Studio Code** | Primary Integrated Development Environment (IDE) on Windows |
| **Pandas** | High-performance data manipulation, filtering, and summary statistics |
| **NumPy** | Array computing, random number generation, and mathematical functions |
| **Scikit-learn** | Pipelines, ColumnTransformers, GridSearchCV, Logistic Regression, Random Forest, metrics |
| **Matplotlib & Seaborn** | Publication-quality statistical visualizations and correlation heatmaps |
| **Joblib** | Model serialization and persistence of fitted pipelines and metadata |
| **Streamlit** | Interactive, browser-based web dashboard for HR analytics and live predictions |
| **Pytest / Unittest** | Automated unit and integration testing suite |

---

## 4. Dataset Description

The system utilizes an HR dataset containing 1,000 employee records with the following schema:

| Feature | Data Type | Domain / Values | Description |
|---|---|---|---|
| `EmployeeID` | String | `EMP0001` - `EMP1000` | Unique employee identifier (excluded from model training) |
| `Age` | Integer | 21 - 60 years | Age of the employee |
| `Department` | String | Sales, Engineering, HR, Marketing, Finance | Operational department unit |
| `MonthlyIncome` | Float | ₹18,000 - ₹220,000 | Monthly gross compensation in Indian Rupees (INR) |
| `YearsAtCompany` | Integer | 0 - 35 years | Total tenure at the company ($\le \text{Age} - 18$) |
| `JobSatisfaction` | Integer | 1 (Low) - 5 (High) | Subjective employee job satisfaction score |
| `WorkLifeBalance` | Integer | 1 (Poor) - 5 (Excellent) | Work-life balance self-assessment rating |
| `PerformanceRating` | Integer | 1 (Low) - 5 (High) | Managerial performance appraisal rating |
| `OverTime` | String | `Yes`, `No` | Whether employee is required to work regular overtime |
| `DistanceFromHome` | Float | 1 - 45 km | Commute distance from home to workplace |
| **`Attrition`** | String | `Yes`, `No` | **Target Variable:** Whether the employee left the company |

---

## 5. Project Directory Structure

```text
Employee-Attrition-Prediction/
│
├── app.py                     # Interactive Streamlit web interface
├── requirements.txt           # Python dependency requirements
├── README.md                  # Comprehensive project documentation
├── VSCODE_SETUP.md            # Windows VS Code setup and troubleshooting guide
├── VIDEO_DEMO_GUIDE.md        # 3-5 minute video walkthrough script
├── LINKEDIN_POST.md           # Professional LinkedIn announcement draft
├── .gitignore                 # Git ignore file for environments and caches
│
├── data/
│   └── employee_data.csv      # Synthetic employee dataset (1,000 records)
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb # Interactive Jupyter notebook for exploration & modeling
│
├── src/
│   ├── __init__.py            # Source package initializer
│   ├── utils.py               # Path definitions, input bounds, and custom exceptions
│   ├── generate_data.py       # Reproducible synthetic data generator
│   ├── preprocessing.py       # Cleaning, ColumnTransformer pipeline & stratified split
│   ├── eda.py                 # Exploratory data analysis & chart generators
│   ├── train_model.py         # Model training, hyperparameter tuning & evaluation
│   └── predict.py             # Model loading, validation & inference service
│
├── models/
│   └── best_model.joblib      # Serialized champion pipeline and metadata
│
├── visuals/
│   ├── attrition_by_department.png   # Attrition rate by department bar chart
│   ├── income_vs_attrition.png       # Monthly income box plot by attrition status
│   ├── correlation_heatmap.png       # Numerical correlation matrix heatmap
│   ├── feature_importance.png        # Champion model feature importance chart
│   ├── attrition_by_overtime.png     # Overtime vs attrition rate comparison
│   └── attrition_by_satisfaction.png # Job satisfaction vs attrition rate trend
│
├── reports/
│   ├── model_comparison.csv          # Cross-validation and test metrics comparison
│   └── best_model_metrics.json       # Champion model configuration & metrics
│
└── tests/
    └── test_project.py        # Automated unit and integration test suite
```

---

## 6. Installation & Setup (Windows / VS Code)

For detailed step-by-step guidance, refer to [VSCODE_SETUP.md](VSCODE_SETUP.md).

### Quick Setup:
1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd "c:\Users\GAYATHRI\OneDrive\Desktop\Employee attrition"
   ```
2. Create and activate a Python 3.11 virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell displays an execution policy restriction, run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*
3. Upgrade `pip` and install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 7. How to Run the Application

All commands must be executed from the root directory:

### Step 1: Generate the Dataset
```powershell
python -m src.generate_data
```
*(Creates `data/employee_data.csv`. Use `--overwrite` to re-generate).*

### Step 2: Run Exploratory Data Analysis (EDA)
```powershell
python -m src.eda
```
*(Prints statistics and exports plots to `visuals/`).*

### Step 3: Train and Evaluate Machine Learning Models
```powershell
python -m src.train_model
```
*(Tunes Logistic Regression and Random Forest, saves comparison metrics and `models/best_model.joblib`).*

### Step 4: Run Real-Time Prediction via CLI
```powershell
python -m src.predict
```
*(Provides an interactive terminal prompt to test individual employee risk).*

### Step 5: Launch the Streamlit Web Dashboard
```powershell
streamlit run app.py
```
*(Opens the interactive web UI in your browser at `http://localhost:8501`).*

### Step 6: Run Automated Tests
```powershell
python -m pytest tests -v
```

---

## 8. Exploratory Data Analysis (EDA) Insights

From the generated dataset and visualizations in `visuals/`:

1. **Overtime Workload (`attrition_by_overtime.png`):** Employees required to work regular overtime exhibit an attrition rate more than **2.5x higher** than non-overtime peers (~32% vs ~11%).
2. **Job Satisfaction (`attrition_by_satisfaction.png`):** Employees rating satisfaction at `1` or `2` account for the highest concentration of voluntary turnover. Attrition decreases monotonically as satisfaction scores rise to `5`.
3. **Compensation Disparity (`income_vs_attrition.png`):** The median monthly income of employees who left is substantially lower than that of retained employees, indicating compensation competitiveness is a key retention factor.
4. **Department Variations (`attrition_by_department.png`):** The **Sales** and **Engineering** departments display higher turnover rates, aligning with industry dynamics of high outbound mobility and competitive market demand.
5. **Correlation Matrix (`correlation_heatmap.png`):** Numerical variables indicate that tenure and age correlate positively with salary, while satisfaction and balance correlate inversely with flight risk.

> *Note: These statistical patterns describe observational associations within the dataset rather than standalone causal factors.*

---

## 9. Model Benchmarking & Performance Comparison

Both models were tuned using 5-Fold Stratified Cross-Validation on the 80% training set (800 records) and evaluated on the held-out 20% test set (200 records, 40 attrition positives):

| Model | Hyperparameters | 5-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** 🏆 | `C=0.1`, `solver='lbfgs'` | **0.5086** | **73.50%** | **0.4110** | **0.7500** | **0.5310** | **0.7641** |
| **Random Forest** | `n_estimators=50`, `max_depth=6`, `min_samples_split=10` | 0.5355 | 75.50% | 0.4151 | 0.5500 | 0.4731 | 0.7417 |

*(Metrics directly populated from `reports/model_comparison.csv` and `reports/best_model_metrics.json`).*

---

## 10. Model Selection Rationale

In employee attrition prediction, identifying an employee who is truly at risk of leaving (**True Positive**) is the primary operational objective. Failing to detect a departing employee (**False Negative**) results in unmitigated replacement costs, project disruptions, and knowledge loss. Conversely, while precision is important to avoid wasting managerial check-in time, a model with higher recall provides superior decision-support value in flight-risk mitigation.

Therefore:
- **Decision Metric:** We optimize for **Test F1-Score** (the harmonic mean of Precision and Recall) rather than raw Accuracy (which is misleading under 80:20 class imbalance).
- **Champion Model:** **Logistic Regression** (with `class_weight='balanced'`) was selected as the champion model. While Random Forest achieved slightly higher test accuracy (75.5% vs 73.5%), Logistic Regression delivered:
  - **Higher Test F1-Score:** `0.5310` vs `0.4731` (+5.8 percentage points)
  - **Significantly Superior Recall:** `0.7500` (75.0%) vs `0.5500` (55.0%), correctly identifying 3 out of 4 departing employees in the test set compared to only 55% for Random Forest.
  - **Higher ROC-AUC:** `0.7641` vs `0.7417`, indicating stronger overall discriminative capability across all decision thresholds.

---

## 11. Governance, Ethics & Limitations

1. **Synthetic Nature:** This project is built on synthetic demonstration data. Results demonstrate algorithmic capabilities and must not be interpreted as validated findings on any specific enterprise workforce.
2. **Supportive Decision Aid:** Predictions are meant as decision-support indicators for proactive check-ins, mentoring, or workload adjustments. They must **never** be used as the basis for adverse employment actions.
3. **Fairness & Bias Auditing:** Real-world HR systems require regular algorithmic fairness audits to ensure disparate impact does not occur across demographic categories.

---

## 12. Future Enhancements

- Integration of SHAP (SHapley Additive exPlanations) for granular individual prediction explanations.
- Temporal sequence modeling incorporating quarterly performance trajectory and promotion velocity.
- Containerization with Docker and deployment via AWS ECS / Azure Container Apps.
- Bias mitigation routines (e.g., Fairlearn) to monitor fairness across demographic attributes.

---

## 13. GitHub Submission Instructions

To push this project to your GitHub account:

1. **Create Repository:** On [github.com](https://github.com/), create a new public repository named `Employee-Attrition-Prediction` (do not initialize with README or .gitignore).
2. **Initialize Git in VS Code:**
   Open PowerShell in the project root and run:
   ```powershell
   git init
   git add .
   git commit -m "feat: complete employee attrition prediction system for Infobharat Interns"
   ```
3. **Connect and Push:**
   ```powershell
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/Employee-Attrition-Prediction.git
   git push -u origin main
   ```
4. **Verify Submission:** Open your GitHub repository URL in the browser and confirm all folders, charts, and documentation are properly visible.

---

## 14. Author Information

- **Author:** [Your Name Here]
- **Internship:** Infobharat Interns
- **Role:** Machine Learning & Data Science Intern
- **GitHub:** [[Your GitHub Profile URL]](https://github.com/)
- **LinkedIn:** [[Your LinkedIn Profile URL]](https://linkedin.com/)
