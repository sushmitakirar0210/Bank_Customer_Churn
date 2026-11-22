# 🏦 Predictive Modeling & Risk Scoring for Bank Customer Churn

An end-to-end Machine Learning and Explainable AI (XAI) system designed to shift retail banking retention from reactive reporting to proactive customer engagement. By engineering behavioral and relationship-strength metrics, benchmarking ensemble classifiers, and integrating SHAP explainability into a multi-tab Streamlit application, this platform assigns calibrated churn risk scores (`0–100%`) and simulates targeted retention interventions.

---

## 📋 Table of Contents
- [Business Context & Objectives](#-business-context--objectives)
- [Project Architecture](#-project-architecture)
- [Feature Engineering & Methodology](#-feature-engineering--methodology)
- [Model Benchmarking & Explainability](#-model-benchmarking--explainability)
- [Streamlit Web Application](#-streamlit-web-application)
- [Installation & Quick Start](#-installation--quick-start)
- [Deliverables](#-deliverables)

---

## 🎯 Business Context & Objectives
Customer attrition directly impacts Customer Lifetime Value (CLV), deposit stability, and cross-sell potential. Traditional churn reporting explains attrition retrospectively, leading to broad, inefficient retention campaigns.

* **Primary Objectives:**
  * Predict customer churn with high discrimination (ROC-AUC) and calibrated probability scores.
  * Identify actionable behavioral drivers of churn rather than relying solely on demographics.
* **Secondary Objectives:**
  * Minimize false positives through dynamic probability threshold optimization.
  * Ensure regulatory transparency and auditability (aligned with European Central Bank / EU AI Act principles) via SHAP explanations.
  * Enable real-time "What-If" retention simulations for relationship managers.

---

## 🗂️ Project Architecture

```text
bank-churn-intelligence/
│
├── data/
│   └── Churn_Modelling.csv              # Retail banking customer dataset (10,000 records)
│
├── models/                              # Generated automatically via src/train.py
│   ├── churn_pipeline.pkl               # Serialized preprocessor + winning ML model
│   ├── shap_explainer.pkl               # Pre-fitted SHAP TreeExplainer
│   ├── feature_names.pkl                # Post-transformation feature names
│   ├── test_shap_values.pkl             # Pre-computed SHAP values for dashboard
│   ├── model_benchmark.csv              # Model evaluation metrics table
│   └── test_predictions.csv             # Validation predictions & churn probabilities
│
├── notebooks/
│   └── 01_eda_and_experiments.ipynb     # Exploratory data analysis & prototyping
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                 # Custom feature engineering & Scikit-Learn pipeline
│   ├── explainability.py                # SHAP global & single-customer explanation helpers
│   └── train.py                         # Stratified CV, benchmarking & artifact export
│
├── reports/
│   ├── research_paper.pdf               # Full analytical research paper
│   ├── executive_summary.pdf            # Policy brief for government/regulatory stakeholders
│   └── figures/                         # Exported ROC, SHAP, and PDP visualizations
│
├── app.py                               # Multi-tab Streamlit web application
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## ⚙️ Feature Engineering & Methodology

All transformations are encapsulated inside a leakage-free Scikit-Learn `Pipeline` (`src/preprocessing.py`) evaluated using an **80/20 Stratified Train-Test Split** and **5-Fold Stratified Cross-Validation**:

| Derived Feature | Formula / Logic | Business Rationale |
| :--- | :--- | :--- |
| `Balance_to_Salary` | `Balance / (EstimatedSalary + 1)` | Captures wealth concentration relative to income. |
| `Product_Density` | `NumOfProducts / (Tenure + 1)` | Measures product adoption velocity over customer lifecycle. |
| `Engagement_Product_Interaction` | `IsActiveMember * NumOfProducts` | Distinguishes active multi-product users from dormant holders. |
| `Age_Tenure_Ratio` | `Tenure / Age` | Reflects life-stage banking loyalty. |
| `Is_Zero_Balance` | `1 if Balance == 0 else 0` | Flags inactive/empty deposit accounts. |

---

## 🧠 Model Benchmarking & Explainability

Five models are trained with class-imbalance handling (`class_weight='balanced'` / `scale_pos_weight`) and evaluated across **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**:

1. **Logistic Regression** *(Interpretability Baseline)*
2. **Decision Tree** *(Rule-Based Non-Linear Benchmark)*
3. **Random Forest** *(Bagged Ensemble)*
4. **Gradient Boosting** *(Sequential Tree Boosting)*
5. **XGBoost** *(Regularized Gradient Boosting Production Candidate)*

### Regulatory-Grade Explainability (SHAP)
* **Global Explainability:** Summary beeswarm plots validate that engagement (`IsActiveMember`), product holding (`NumOfProducts`), and account balance dominate predictions over protected attributes (`Gender`).
* **Local Explainability:** Waterfall plots break down individual customer risk scores into positive and negative feature contributions.

---

## 🖥️ Streamlit Web Application

The interactive dashboard (`app.py`) consists of four operational modules:
1. **🎯 Customer Risk Calculator:** Input customer attributes to compute a live churn probability gauge (`Low`, `Medium`, or `High` risk tier) paired with a local SHAP waterfall explanation.
2. **🔄 What-If Scenario Simulator:** Select high-risk customers and simulate retention levers (e.g., reactivating membership, adjusting product counts) to observe *Before vs. After* churn probability shifts.
3. **📊 Portfolio Risk Distribution:** Visualize predicted probability distributions across geographies and tune the campaign decision threshold to monitor Precision, Recall, and False Positives in real time.
4. **🧠 Explainability & Benchmarks:** Inspect the full model comparison matrix and global SHAP driver rankings.

---

## 🚀 Installation & Quick Start

### 1. Clone & Set Up Environment
*(Python 3.11–3.13 recommended for pre-built binary wheels on Windows/macOS/Linux.)*

```bash
git clone <your-repository-url>
cd bank-churn-intelligence
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train Models & Generate Artifacts
Ensure `Churn_Modelling.csv` is placed inside the `data/` directory, then run:
```bash
python -m src.train
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📄 Deliverables
* **Interactive Analytics Dashboard (`app.py`)** — Live churn scoring, SHAP explainability, and scenario simulation.
* **Research Paper (`reports/research_paper.pdf`)** — Comprehensive EDA, feature engineering ablation, model evaluation, and retention strategy recommendations.
* **Executive Summary (`reports/executive_summary.pdf`)** — Concise governance and policy brief for regulatory and banking stakeholders.
