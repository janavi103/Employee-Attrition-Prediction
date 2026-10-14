# Video Demonstration Recording Guide (3–5 Minutes)

This step-by-step walkthrough is designed for recording an impressive, professional 3-to-5 minute video demonstration of your **Employee Attrition Prediction System** for your **Infobharat Interns** internship submission.

---

## 🛠️ Recording Setup & Tools

- **Tool Recommendation:**
  - Built-in Windows Snipping Tool / Xbox Game Bar (`Win + Alt + R` to start/stop recording).
  - Or free desktop screen recorders like **OBS Studio** or **ShareX**.
- **Display Resolution:** Set display to 1080p (1920x1080) for crisp text.
- **Audio:** Use a clear microphone; keep background noise minimal.
- **Preparation:**
  1. Open VS Code with your project folder: `Employee attrition`.
  2. Pre-open a browser window at `http://localhost:8501` (running `streamlit run app.py`).
  3. Ensure terminal has completed `python -m src.train_model` and `python -m pytest tests -v`.

---

## ⏱️ Video Timeline & Script Blueprint

### Minute 0:00 - 0:45 | Introduction & Business Problem
- **What to Show:** VS Code workspace showing `README.md` and project file explorer.
- **Talking Points:**
  - *"Hello everyone, my name is [Your Name], and I am presenting my internship project for Infobharat Interns: the Employee Attrition Prediction System."*
  - *"Employee turnover is a major cost and knowledge-loss factor for enterprises. In this project, we built an end-to-end Machine Learning pipeline that analyzes factors like compensation, overtime, commute distance, and job satisfaction to predict flight risk and empower HR teams with proactive retention insights."*

### Minute 0:45 - 1:30 | Dataset & Exploratory Data Analysis (EDA)
- **What to Show:** Terminal running `python -m src.eda`, followed by displaying generated charts from the `visuals/` folder in VS Code or the Streamlit EDA tab.
- **Talking Points:**
  - *"Our dataset contains 1,000 employee records with 9 key features, including demographics, salary, satisfaction scores, and overtime."*
  - *"Looking at our EDA charts:*
    - *Department Attrition: Sales and Engineering show dynamic flight risks.*
    - *Overtime Impact: Employees working overtime show significantly higher attrition rates.*
    - *Salary Distribution: Lower compensation brackets correlate with increased turnover.*
    - *Job Satisfaction: Clear inverse relationship between satisfaction scores (1 to 5) and attrition rate."*
  - *"Note that these represent statistical associations rather than standalone causal drivers."*

### Minute 1:30 - 2:30 | Model Training & Evaluation
- **What to Show:** Open `reports/model_comparison.csv` or Streamlit's **Model Evaluation** tab.
- **Talking Points:**
  - *"To ensure high generalization, we built a Scikit-learn Pipeline with ColumnTransformer, median/mode imputers, StandardScaler, and One-Hot Encoding fitted strictly on training data to prevent data leakage."*
  - *"We compared two algorithms using 5-fold cross-validated hyperparameter tuning:*
    - *1. Logistic Regression (optimized across C and solvers with balanced class weighting)*
    - *2. Random Forest Classifier (tuned across estimators, max_depth, and min_samples_split)*
  - *"Because attrition exhibits class imbalance, we selected our champion model based on the Test F1-Score, ensuring a strong balance between catching flight risks (Recall) and minimizing false alarms (Precision)."*
  - *Show the confusion matrix and highlight the feature importance chart (`visuals/feature_importance.png`).*

### Minute 2:30 - 4:00 | Live Interactive Streamlit Demo
- **What to Show:** The Streamlit dashboard running in browser (`http://localhost:8501`).
- **Actions to Perform:**
  1. **High-Risk Employee Test:**
     - Set Age: `27`, Department: `Sales`, Income: `₹35,000`, Years at Company: `1`.
     - OverTime: `Yes`, Distance: `32 km`, Job Satisfaction: `1`, Work-Life Balance: `1`.
     - Click **Predict Attrition**.
     - Point out the red warning box: **"Likely to Leave"**, probability bar showing high flight risk (e.g., ~80%+).
  2. **Low-Risk Employee Test:**
     - Set Age: `45`, Department: `Engineering`, Income: `₹95,000`, Years at Company: `7`.
     - OverTime: `No`, Distance: `6 km`, Job Satisfaction: `4`, Work-Life Balance: `4`.
     - Click **Predict Attrition**.
     - Point out the green card: **"Likely to Stay"**, low attrition probability.
  3. Show the **Governance & Ethics** tab:
     - Emphasize that predictions are decision-support aids for supportive HR retention, not for automated punitive employment decisions.

### Minute 4:00 - 4:45 | Code Structure, Testing & Wrap-Up
- **What to Show:** Switch to VS Code terminal.
- **Actions to Perform:**
  - Run `python -m pytest tests -v` to show all automated tests passing.
  - Show the modular structure (`src/`, `models/`, `reports/`, `visuals/`).
- **Closing Script:**
  - *"All unit tests pass cleanly, covering schema validation, preprocessing pipelines, inference bounds, and domain restrictions."*
  - *"Thank you Infobharat Interns for this valuable opportunity. The complete source code is available in my GitHub repository linked in the description."*

---

## 📌 Checklist Before Recording

- [ ] Run `python -m src.generate_data`
- [ ] Run `python -m src.eda`
- [ ] Run `python -m src.train_model`
- [ ] Verify `streamlit run app.py` opens cleanly without errors
- [ ] Clean up terminal history and zoom VS Code text font (`Ctrl + +`) for legibility
