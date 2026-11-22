import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from src.explainability import explain_single_customer
import streamlit as st

st.set_page_config(
    page_title="Bank Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("models/churn_pipeline.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    test_preds = pd.read_csv("models/test_predictions.csv")
    benchmark = pd.read_csv("models/model_benchmark.csv")
    test_shap = joblib.load("models/test_shap_values.pkl")
    return pipeline, explainer, feature_names, test_preds, benchmark, test_shap


try:
    pipeline, explainer, feature_names, test_preds, benchmark, test_shap = (
        load_artifacts()
    )
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `py -3.13 -m src.train` in your terminal"
        " first."
    )
    st.stop()


def get_risk_tier(prob):
    if prob < 0.30:
        return "Low Risk", "green"
    elif prob < 0.65:
        return "Medium Risk", "orange"
    return "High Risk", "red"


st.title("🏦 Predictive Modeling & Risk Scoring for Bank Customer Churn")
st.markdown(
    "Early churn risk identification, behavioral driver explainability, and"
    " retention scenario simulation."
)

tab1, tab2, tab3, tab4 = st.tabs((
    "🎯 Customer Risk Calculator",
    "🔄 What-If Scenario Simulator",
    "📊 Portfolio Risk Distribution",
    "🧠 Explainability & Benchmarks",
))

# ==========================================
# TAB 1: CUSTOMER CHURN RISK CALCULATOR
# ==========================================
with tab1:
    st.subheader("Individual Customer Risk Scoring & Local SHAP Explanation")
    col_input, col_result = st.columns((1, 1.4), gap="large")

    with col_input:
        st.markdown("#### Customer Profile Inputs")
        c1, c2 = st.columns(2)
        with c1:
            geography = st.selectbox("Geography", ("France", "Germany", "Spain"))
            gender = st.selectbox("Gender", ("Female", "Male"))
            age = st.slider("Age", 18, 92, 42)
            tenure = st.slider("Tenure (Years)", 0, 10, 5)
            credit_score = st.number_input("Credit Score", 300, 850, 650)
        with c2:
            balance = st.number_input(
                "Account Balance ($)", 0.0, 250000.0, 75000.0, step=1000.0
            )
            salary = st.number_input(
                "Estimated Salary ($)", 0.0, 250000.0, 85000.0, step=1000.0
            )
            num_products = st.selectbox("Number of Products", (1, 2, 3, 4))
            has_cr_card = st.selectbox(
                "Has Credit Card?",
                (1, 0),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            is_active = st.selectbox(
                "Is Active Member?",
                (1, 0),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        customer_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary,
        }])

    with col_result:
        prob = float(pipeline.predict_proba(customer_df).item(1))
        tier, color = get_risk_tier(prob)

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 42}},
                title={"text": f"Churn Risk Score — <b>{tier}</b>"},
                gauge={
                    "axis": {"range": (0, 100)},
                    "bar": {"color": color},
                    "steps": (
                        {"range": (0, 30), "color": "rgba(0, 200, 83, 0.15)"},
                        {"range": (30, 65), "color": "rgba(255, 165, 0, 0.15)"},
                        {"range": (65, 100), "color": "rgba(255, 0, 0, 0.15)"},
                    ),
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, width="stretch")

        st.markdown("#### Why did the model assign this score? (SHAP Waterfall)")
        shap_single = explain_single_customer(
            pipeline, explainer, feature_names, customer_df
        )
        fig_wf, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_single, max_display=8, show=False)
        st.pyplot(fig_wf, clear_figure=True)

# ==========================================
# TAB 2: WHAT-IF SCENARIO SIMULATOR
# ==========================================
with tab2:
    st.subheader("Retention Action Simulator (Before vs. After Intervention)")
    st.markdown(
        "Select a high-risk customer from the validation portfolio and test"
        " targeted retention interventions."
    )

    high_risk_pool = test_preds[
        test_preds["Churn_Probability"] >= 0.50
    ].reset_index(drop=True)
    selected_idx = st.selectbox(
        "Select High-Risk Customer Index",
        high_risk_pool.index,
        format_func=lambda idx: (
            f"Customer #{idx} | Age: {int(high_risk_pool.loc[idx, 'Age'])} |"
            f" {high_risk_pool.loc[idx, 'Geography']} | Baseline Risk:"
            f" {high_risk_pool.loc[idx, 'Churn_Probability']:.1%}"
        ),
    )

    base_row = high_risk_pool.loc[[selected_idx]].drop(
        columns=["Actual_Exited", "Churn_Probability"]
    )
    base_prob = float(pipeline.predict_proba(base_row).item(1))

    st.markdown("#### Adjust Retention Levers")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        sim_active = st.selectbox(
            "Target Activity Status",
            (0, 1),
            index=int(base_row["IsActiveMember"].values.item(0)),
            format_func=lambda x: "Active (1)" if x == 1 else "Inactive (0)",
        )
    with s2:
        sim_products = st.selectbox(
            "Target Product Count",
            (1, 2, 3, 4),
            index=int(base_row["NumOfProducts"].values.item(0)) - 1,
        )
    with s3:
        sim_balance = st.number_input(
            "Simulated Balance ($)",
            0.0,
            250000.0,
            float(base_row["Balance"].values.item(0)),
            step=5000.0,
        )
    with s4:
        sim_card = st.selectbox(
            "Credit Card Cross-Sell",
            (0, 1),
            index=int(base_row["HasCrCard"].values.item(0)),
            format_func=lambda x: "Has Card (1)" if x == 1 else "No Card (0)",
        )

    sim_row = base_row.copy()
    sim_row["IsActiveMember"] = sim_active
    sim_row["NumOfProducts"] = sim_products
    sim_row["Balance"] = sim_balance
    sim_row["HasCrCard"] = sim_card

    new_prob = float(pipeline.predict_proba(sim_row).item(1))
    delta = new_prob - base_prob

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Churn Probability", f"{base_prob:.1%}")
    m2.metric(
        "Post-Intervention Probability",
        f"{new_prob:.1%}",
        delta=f"{delta:.1%}",
        delta_color="inverse",
    )
    m3.metric("Updated Risk Tier", get_risk_tier(new_prob)[0])

# ==========================================
# TAB 3: PORTFOLIO PROBABILITY DISTRIBUTION
# ==========================================
with tab3:
    st.subheader("Portfolio Churn Probability Distribution & Threshold Tuning")

    threshold = st.slider(
        "Decision Threshold for Retention Campaign Trigger", 0.10, 0.90, 0.50, 0.05
    )

    preds_binary = (test_preds["Churn_Probability"] >= threshold).astype(int)
    tp = ((preds_binary == 1) & (test_preds["Actual_Exited"] == 1)).sum()
    fp = ((preds_binary == 1) & (test_preds["Actual_Exited"] == 0)).sum()
    fn = ((preds_binary == 0) & (test_preds["Actual_Exited"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Customers Flagged for Outreach",
        f"{preds_binary.sum()} / {len(test_preds)}",
    )
    k2.metric("Expected Precision", f"{precision:.1%}")
    k3.metric("Expected Recall (Churn Captured)", f"{recall:.1%}")
    k4.metric("False Alarms (False Positives)", f"{fp}")

    fig_hist = px.histogram(
        test_preds,
        x="Churn_Probability",
        color="Geography",
        nbins=40,
        barmode="overlay",
        title="Predicted Churn Probability Distribution by Geography",
        labels={"Churn_Probability": "Predicted Churn Probability"},
    )
    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
    )
    st.plotly_chart(fig_hist, width="stretch")

# ==========================================
# TAB 4: GLOBAL EXPLAINABILITY & BENCHMARKS
# ==========================================
with tab4:
    st.subheader("Model Performance Comparison & Global Feature Drivers")

    st.markdown(
        "#### 1. Model Benchmarking Summary (Stratified 5-Fold CV & Test Set)"
    )
    st.dataframe(benchmark, width="stretch")

    st.markdown(
        "#### 2. Global SHAP Feature Importance (Behavioral vs. Demographic)"
    )
    fig_shap, ax = plt.subplots(figsize=(10, 5))
    vals = test_shap[:, :, 1] if len(test_shap.shape) == 3 else test_shap
    shap.summary_plot(vals, show=False, max_display=12)
    st.pyplot(fig_shap, clear_figure=True)
