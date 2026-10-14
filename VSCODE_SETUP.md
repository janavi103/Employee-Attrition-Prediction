# Visual Studio Code & Environment Setup Guide (Windows)

This guide provides beginner-friendly, step-by-step instructions for configuring and executing the **Employee Attrition Prediction System** inside Visual Studio Code (VS Code) on Windows 10/11.

---

## 1. Prerequisites Installation

### A. Python 3.11
1. Download Python 3.11 from the official portal: [python.org/downloads](https://www.python.org/downloads/).
2. Launch the installer and **ensure you check**:
   - `[x] Add python.exe to PATH`
   - `[x] Use admin privileges when installing py.exe`
3. Verify your Python installations by opening PowerShell and running:
   ```powershell
   py --list
   ```
   *Expected output: lists installed versions such as `-V:3.11`.*

### B. Visual Studio Code
1. Download and install VS Code from [code.visualstudio.com](https://code.visualstudio.com/).
2. Open VS Code, open the Extensions tab (`Ctrl+Shift+X`), and install:
   - **Python** (by Microsoft)
   - **Pylance** (by Microsoft)
   - **Jupyter** (by Microsoft)

---

## 2. Opening the Project in VS Code

1. Launch VS Code.
2. Click **File** → **Open Folder...** (`Ctrl+K Ctrl+O`).
3. Navigate to and select the project folder:
   ```
   c:\Users\GAYATHRI\OneDrive\Desktop\Employee attrition
   ```
4. Open the integrated terminal:
   - Menu: **Terminal** → **New Terminal** (`Ctrl+` `~`)
   - Confirm that the terminal path ends at the project root (`Employee attrition`).

---

## 3. Creating & Activating Virtual Environment

Using an isolated virtual environment guarantees that your packages and Python versions do not conflict with other system libraries.

### Step 3.1: Create `.venv`
Run the following command in PowerShell from the project root:
```powershell
py -3.11 -m venv .venv
```
*(If `py -3.11` is not recognized, run `py --list` to view your installed versions, or run `python -m venv .venv` if Python 3.11 is your default interpreter).*

### Step 3.2: Activate the Virtual Environment
In PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(You will observe `(.venv)` prefixed to your command prompt).*

> **Troubleshooting PowerShell Script Execution:**  
> If PowerShell displays an error stating:  
> `File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system.`  
> Run this safe, user-scoped command to permit local scripts without modifying system-wide security:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Alternatively, switch your terminal to Command Prompt (`cmd`) and activate:
> ```cmd
> .venv\Scripts\activate.bat
> ```

---

## 4. Installing Dependencies

Upgrade `pip` and install all necessary libraries:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Selecting the Python Interpreter in VS Code

To ensure VS Code's editor, linters, and test runner utilize your virtual environment:
1. Press `Ctrl+Shift+P` to open the Command Palette.
2. Type and select: `Python: Select Interpreter`.
3. Choose the interpreter pointing to your virtual environment:
   ```
   Python 3.11.x ('.venv': venv) .\.venv\Scripts\python.exe
   ```

---

## 6. Running the Application Workflow

All commands must be executed from the project root in your terminal:

### Step 6.1: Generate Synthetic Employee Dataset
```powershell
python -m src.generate_data
```
- Creates `data/employee_data.csv` containing 1,000 employee records with realistic organizational dynamics.
- Note: It will not overwrite an existing dataset unless called with `--overwrite`.

### Step 6.2: Run Exploratory Data Analysis (EDA)
```powershell
python -m src.eda
```
- Outputs statistical summaries to the terminal.
- Generates 5 high-resolution charts in `visuals/`.

### Step 6.3: Train, Tune & Evaluate Machine Learning Models
```powershell
python -m src.train_model
```
- Executes 5-fold cross-validated hyperparameter tuning on Logistic Regression and Random Forest.
- Evaluates models on the 20% stratified test set.
- Selects the champion model based on test F1-score.
- Saves:
  - `reports/model_comparison.csv`
  - `reports/best_model_metrics.json`
  - `visuals/feature_importance.png`
  - `models/best_model.joblib`

### Step 6.4: Interactive Terminal Prediction (Optional)
```powershell
python -m src.predict
```
- Allows manual testing or evaluating sample employee risk profiles directly in the console.

### Step 6.5: Launch the Interactive Streamlit Web Interface
```powershell
streamlit run app.py
```
- Automatically opens your default web browser at `http://localhost:8501`.
- Explore live employee risk scoring, interactive plots, model metrics, and ethical governance guidelines.
- To stop the server, press `Ctrl+C` in the terminal.

---

## 7. Running Automated Test Suite

To verify code integrity, input sanitization, and model robustness:
```powershell
python -m pytest tests -v
```
All tests should pass with green checkmarks.

---

## 8. Common Troubleshooting Q&A

| Issue / Error | Root Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Running a script from inside `src/` rather than project root | Run from project root using `python -m src.<module_name>` |
| `Activate.ps1 cannot be loaded` | PowerShell ExecutionPolicy default restriction | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `Port 8501 is already in use` | Another Streamlit process is running | Run `streamlit run app.py --server.port 8502` |
| VS Code underlines imports in yellow | VS Code interpreter not set to `.venv` | Press `Ctrl+Shift+P` → `Python: Select Interpreter` → select `.venv` |
| Model file not found in Streamlit | Training step was skipped | Run `python -m src.train_model` before launching `streamlit run app.py` |
