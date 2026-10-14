"""Streamlit Web Interface for the Employee Attrition Prediction System.

Provides an interactive dashboard displaying organizational metrics,
EDA visualizations, model comparison results, and a real-time prediction
form for individual employee attrition risk analysis.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.preprocessing import TARGET_MAPPING
from src.utils import (
    COMPARISON_FILE,
    DATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    PROJECT_ROOT,
    VALID_DEPARTMENTS,
    VALID_OVERTIME,
    VISUALS_DIR,
    ModelError,
    ValidationError,
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, professional typography and cards
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-box-leave {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1.2rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
    .risk-box-stay {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 1.2rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_predictor():
    """Cache the predictor instance to avoid redundant loading across interactions."""
    if not MODEL_FILE.exists():
        return None
    try:
        from src.predict import AttritionPredictor

        return AttritionPredictor(MODEL_FILE)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_dataset_summary():
    """Load cached dataset metrics."""
    if not DATA_FILE.exists():
        return None
    try:
        df = pd.read_csv(DATA_FILE)
        total = len(df)
        left = (df["Attrition"] == "Yes").sum()
        rate = (left / total) * 100 if total > 0 else 0
        return {"total": total, "left": left, "rate": rate, "df": df}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_reports():
    """Load model comparison table and best model metrics."""
    comp_df = pd.read_csv(COMPARISON_FILE) if COMPARISON_FILE.exists() else None
    metrics_data = None
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = None
    return comp_df, metrics_data


def render_header():
    """Render top header and application badges."""
    st.markdown('<div class="main-title">👥 Employee Attrition Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A machine learning decision-support system analyzing workplace factors, '
        "satisfaction, and compensation to estimate employee retention probability.</div>",
        unsafe_allow_html=True,
    )


def render_kpis(summary, metrics_data):
    """Render summary KPI cards."""
    col1, col2, col3, col4 = st.columns(4)

    total_emp = f"{summary['total']:,}" if summary else "N/A"
    att_rate = f"{summary['rate']:.1f}%" if summary else "N/A"
    champion_model = metrics_data.get("model_name", "N/A") if metrics_data else "Not Trained"
    f1_val = (
        f"{metrics_data['test_metrics']['f1_score']:.3f}"
        if metrics_data and "test_metrics" in metrics_data
        else "N/A"
    )

    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{total_emp}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Baseline Attrition</div>
                <div class="metric-value">{att_rate}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Champion Model</div>
                <div class="metric-value" style="font-size:1.3rem;">{champion_model}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Test F1-Score</div>
                <div class="metric-value">{f1_val}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.write("")


def render_prediction_tab(predictor):
    """Render the employee prediction form and output cards."""
    st.subheader("🔍 Individual Employee Attrition Prediction")
    st.write(
        "Enter employee profile attributes below to calculate the predicted attrition status "
        "and associated confidence probabilities."
    )

    if predictor is None:
        st.warning(
            "⚠️ Trained model artifact (`models/best_model.joblib`) not found. "
            "Please train the model first by executing: `python -m src.train_model` in your terminal."
        )
        return

    with st.form("attrition_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (Years)", min_value=18, max_value=70, value=30, step=1)
            dept = st.selectbox("Department", options=VALID_DEPARTMENTS, index=0)
            income = st.number_input(
                "Monthly Income (₹ INR)",
                min_value=10000.0,
                max_value=500000.0,
                value=45000.0,
                step=1000.0,
                format="%.0f",
            )

        with col2:
            max_tenure = max(0, age - 18)
            tenure = st.number_input(
                "Years at Company",
                min_value=0,
                max_value=max_tenure if max_tenure > 0 else 1,
                value=min(3, max_tenure),
                step=1,
            )
            ot = st.selectbox("Works Overtime?", options=VALID_OVERTIME, index=0)
            dist = st.number_input(
                "Distance from Home (km)",
                min_value=0.0,
                max_value=150.0,
                value=12.0,
                step=1.0,
            )

        with col3:
            js = st.slider("Job Satisfaction", min_value=1, max_value=5, value=3)
            wlb = st.slider("Work-Life Balance", min_value=1, max_value=5, value=3)
            perf = st.slider("Performance Rating", min_value=1, max_value=5, value=3)

        submitted = st.form_submit_button("Predict Attrition", use_container_width=True, type="primary")

    if submitted:
        input_data = {
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

        try:
            result = predictor.predict_single(input_data)
            status = result["prediction_status"]
            att_prob = result["attrition_probability_pct"]
            stay_prob = result["stay_probability_pct"]
            pred_class = result["predicted_attrition"]

            st.write("---")
            st.subheader("📊 Prediction Results")

            r_col1, r_col2 = st.columns([1, 1])

            with r_col1:
                if pred_class == "Yes":
                    st.markdown(
                        f"""<div class="risk-box-leave">
                            <h3 style="color:#b91c1c; margin:0;">⚠️ Status: {status}</h3>
                            <p style="margin:5px 0 0 0; color:#374151;">
                                The model predicts an elevated likelihood of this employee leaving the organization.
                            </p>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div class="risk-box-stay">
                            <h3 style="color:#15803d; margin:0;">✅ Status: {status}</h3>
                            <p style="margin:5px 0 0 0; color:#374151;">
                                The model predicts this employee is likely to remain with the organization.
                            </p>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.caption(f"**Employed Model:** {result['model_name']}")

            with r_col2:
                st.write(f"**Attrition Probability:** `{att_prob}%`")
                st.progress(float(att_prob) / 100.0)

                st.write(f"**Retention Probability:** `{stay_prob}%`")
                st.progress(float(stay_prob) / 100.0)

            st.info(f"💡 **HR Guidance:** {result['disclaimer']}")

        except ValidationError as ve:
            st.error(f"❌ Input Validation Error: {ve}")
        except Exception as e:
            st.error(f"❌ Prediction Failed: {e}")


def render_eda_tab():
    """Render EDA charts and statistical commentary."""
    st.subheader("📈 Exploratory Data Analysis & Visualizations")
    st.write(
        "Interactive display of exploratory data visualizations generated from organizational HR data."
    )

    plot_files = [
        ("attrition_by_department.png", "Attrition Rate by Department", "Proportion of employees leaving across organizational units."),
        ("income_vs_attrition.png", "Monthly Income Distribution vs Attrition", "Comparison of salary distributions between retained employees and those who left."),
        ("attrition_by_overtime.png", "Attrition Rate by Overtime Requirement", "Attrition variance between employees required to work overtime versus regular schedules."),
        ("attrition_by_satisfaction.png", "Attrition Rate by Job Satisfaction Score", "Retention trends across subjective satisfaction ratings on a 1-5 scale."),
        ("correlation_heatmap.png", "Numerical Feature Correlation Matrix", "Pairwise Pearson correlation of numerical factors and binary attrition (association, not causation)."),
        ("feature_importance.png", "Model Feature Importance", "Relative weight and impact of factors in the trained champion classification model."),
    ]

    cols = st.columns(2)
    for i, (fname, title, desc) in enumerate(plot_files):
        img_path = VISUALS_DIR / fname
        with cols[i % 2]:
            st.markdown(f"##### {title}")
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
                st.caption(desc)
            else:
                st.warning(f"Chart `{fname}` has not been generated yet. Run: `python -m src.eda`.")
            st.write("")


def render_models_tab(comp_df, metrics_data):
    """Render model performance comparison table and metrics."""
    st.subheader("🤖 Machine Learning Model Benchmarking")
    st.write(
        "Comparison of candidate models evaluated on an isolated stratified test set (20%) "
        "following hyperparameter optimization via 5-fold cross-validation."
    )

    if comp_df is not None:
        st.markdown("#### Test Set Performance Metrics")
        # Style dataframe for presentation
        styled_df = comp_df.copy()
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Model comparison report (`reports/model_comparison.csv`) not found. Run: `python -m src.train_model`.")

    if metrics_data:
        st.markdown("#### Champion Model Specifications")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write(f"- **Selected Model:** `{metrics_data.get('model_name')}`")
            st.write(f"- **Selection Rule:** {metrics_data.get('selection_rule')}")
            st.write(f"- **Best Hyperparameters:**")
            st.json(metrics_data.get("best_hyperparameters", {}))
        with col_m2:
            st.write("- **Detailed Classification Report:**")
            clf_rep = metrics_data.get("test_metrics", {}).get("classification_report", {})
            if clf_rep:
                rep_df = pd.DataFrame(clf_rep).transpose()
                st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)

            cm = metrics_data.get("test_metrics", {}).get("confusion_matrix", None)
            if cm:
                st.write(f"- **Confusion Matrix [ [TN, FP], [FN, TP] ]:** `{cm}`")


def render_governance_tab():
    """Render limitations, ethics, and HR governance statement."""
    st.subheader("⚖️ Governance, Limitations & Ethical Considerations")
    st.markdown(
        """
        ### 1. Synthetic Data Disclaimer
        This application is constructed using a synthetic demonstration dataset designed to reflect realistic 
        organizational HR dynamics. The synthetic nature means that while mathematical properties (correlations, 
        imbalances, interactions) are authentic, predictions **do not** reflect any actual corporate employee base.

        ### 2. Correlation vs. Causation
        Statistical associations identified in EDA (e.g., higher attrition among overtime workers or lower-paid employees) 
        indicate **observational correlation, not direct causality**. Management interventions should investigate underlying 
        systemic factors rather than treating single model variables in isolation.

        ### 3. Ethical HR Guidance
        - **Supportive Decision Aid:** Predictive outputs are intended strictly as early indicators for preventative retention 
          interventions (e.g., career counseling, workload rebalancing, satisfaction reviews).
        - **Prohibition on Automated Adverse Action:** Model scores must **never** be used as the sole basis for promotion, 
          compensation denial, severance, or disciplinary decisions.
        - **Bias Monitoring:** Production HR machine learning models must undergo periodic audits for disparate impact 
          across protected demographic categories.
        """
    )


def main():
    render_header()

    # Load resources
    predictor = load_model_predictor()
    summary = load_dataset_summary()
    comp_df, metrics_data = load_reports()

    # Display KPIs
    render_kpis(summary, metrics_data)

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔮 Live Prediction",
            "📊 Exploratory Data Analysis",
            "🏆 Model Evaluation",
            "🛡️ Governance & Ethics",
        ]
    )

    with tab1:
        render_prediction_tab(predictor)
    with tab2:
        render_eda_tab()
    with tab3:
        render_models_tab(comp_df, metrics_data)
    with tab4:
        render_governance_tab()


if __name__ == "__main__":
    main()
