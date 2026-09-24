import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Import ML Engine ────────────────────────────────────────────────────────
import ml_engine

# ── Streamlit Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Insurance Fraud Analytics Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS & Styling ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Gradient Header */
    .hero-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00d2ff 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.92;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* Metric & KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #00bcd4;
        margin-top: 0.3rem;
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.75;
    }

    /* Content Cards */
    .custom-card {
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .custom-card-header {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* Prediction Result Cards */
    .verdict-box {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
        animation: fadeIn 0.4s ease-in-out;
    }
    .verdict-high {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.12) 0%, rgba(192, 57, 43, 0.05) 100%);
        border: 2px solid #e74c3c;
    }
    .verdict-low {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.12) 0%, rgba(39, 174, 96, 0.05) 100%);
        border: 2px solid #2ecc71;
    }
    .verdict-badge {
        display: inline-block;
        padding: 0.45rem 1.6rem;
        font-weight: 800;
        font-size: 1.2rem;
        border-radius: 50px;
        letter-spacing: 0.06em;
        color: white;
        margin-bottom: 1rem;
    }
    .verdict-badge.high { background-color: #e74c3c; }
    .verdict-badge.low { background-color: #2ecc71; }

    .score-number {
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .score-number.high { color: #e74c3c; }
    .score-number.low { color: #2ecc71; }

    .action-badge {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.35rem 1rem;
        background: rgba(128, 128, 128, 0.1);
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Algorithm Tag */
    .algo-tag {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        background: #00bcd4;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ── Load and Cache ML Pipeline and Data ─────────────────────────────────────
@st.cache_resource(show_spinner="Training and benchmarks running across all algorithms...")
def get_ml_suite():
    return ml_engine.train_all_models()

@st.cache_data(show_spinner="Loading Dataset...")
def get_dataset():
    return ml_engine.load_and_clean_data()

try:
    ml_data = get_ml_suite()
    raw_df, clean_df = get_dataset()
    pipelines = ml_data['pipelines']
    comp_df = ml_data['comparison_df']
    cm_dict = ml_data['confusion_matrices']
    roc_dict = ml_data['roc_data']
    reports_dict = ml_data['classification_reports']
except Exception as e:
    st.error(f"⚠️ Error initializing data and models: {e}")
    st.stop()

# ── Sidebar Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.3rem;">🛡️</div>
            <h2 style="margin: 0; font-weight: 800; font-size: 1.4rem;">FraudShield AI</h2>
            <p style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.2rem;">Insurance Fraud Intelligence Suite</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation",
        [
            "🏠 Overview & Pipeline",
            "📊 Exploratory Data Analysis (EDA)",
            "🤖 All Algorithms & Model Comparison",
            "🛡️ Interactive Fraud Risk Predictor",
            "📂 Batch Claims Predictor (CSV)",
            "📑 Semester 5 Project Documentation"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Quick Model Stats")
    st.write(f"• **Cleaned Records:** `{clean_df.shape[0]:,}`")
    st.write(f"• **Features:** `{len(ml_data['feature_cols'])}`")
    st.write(f"• **Algorithms Trained:** `5`")
    st.write("• **Best Model:** `Random Forest (Tuned)`")
    
    st.markdown("---")
    st.caption("Vehicle Insurance Fraud Detection System | Machine Learning Sem-5 Project")

# ═════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW & PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
if app_mode == "🏠 Overview & Pipeline":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">🛡️ Vehicle Insurance Fraud Detection</h1>
            <p class="hero-subtitle">Comprehensive Machine Learning Pipeline & Multi-Algorithm Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Raw Claims Ingested</div>
                <div class="kpi-value">12,002</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Cleaned Dataset Records</div>
                <div class="kpi-value">7,640</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Fraud Prevalence Rate</div>
                <div class="kpi-value" style="color:#e74c3c;">25.79%</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Algorithms Benchmarked</div>
                <div class="kpi-value" style="color:#2ecc71;">5 Models</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([6, 5])
    
    with c_left:
        st.markdown("""
            <div class="custom-card">
                <div class="custom-card-header">🎯 Project Objective & Problem Statement</div>
                <p style="line-height: 1.7;">
                    Insurance fraud costs the auto insurance industry billions annually, leading to increased premiums for honest policyholders.
                    The primary objective of this project is to build an automated, high-precision supervised machine learning system to classify 
                    incoming insurance claims as <b>Genuine (0 / N)</b> or <b>Fraudulent (1 / Y)</b> before settlement disbursements.
                </p>
                <div style="margin-top: 1rem;">
                    <b>Key Project Pillars:</b>
                    <ul style="line-height: 1.8; margin-top: 0.5rem;">
                        <li><b>Imbalanced Data Handling:</b> Cost-sensitive learning and balanced class weights to maximize fraud recall.</li>
                        <li><b>Data Hygiene & Preprocessing:</b> Median/mode imputation, standard scaling, and IQR outlier containment.</li>
                        <li><b>Multi-Model Comparison:</b> Rigorous 5-fold cross-validation across linear, tree, bagging, and boosting algorithms.</li>
                        <li><b>Hyperparameter Optimization:</b> GridSearchCV tuning for optimal tree depth, estimators, and splits.</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with c_right:
        st.markdown("""
            <div class="custom-card">
                <div class="custom-card-header">🔄 End-to-End ML Architecture</div>
                <div style="display:flex; flex-direction:column; gap:0.8rem; font-size:0.95rem;">
                    <div style="padding:0.75rem; background:rgba(0,188,212,0.1); border-left:4px solid #00bcd4; border-radius:4px;">
                        <b>1. Ingestion & Cleaning:</b> Standardize schema, impute missing values, handle '*' placeholders, IQR filter.
                    </div>
                    <div style="padding:0.75rem; background:rgba(52,152,219,0.1); border-left:4px solid #3498db; border-radius:4px;">
                        <b>2. Preprocessing Pipeline:</b> <code>StandardScaler</code> for continuous variables + <code>OrdinalEncoder</code> for categories.
                    </div>
                    <div style="padding:0.75rem; background:rgba(155,89,182,0.1); border-left:4px solid #9b59b6; border-radius:4px;">
                        <b>3. Modeling & Evaluation:</b> 5 algorithms (Logistic Regression, Decision Tree, Random Forest, AdaBoost, Tuned RF).
                    </div>
                    <div style="padding:0.75rem; background:rgba(46,204,113,0.1); border-left:4px solid #2ecc71; border-radius:4px;">
                        <b>4. Live Decision Support:</b> Real-time fraud risk scoring, SIU referrals, factor explanations & CSV batch audits.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Weekly Progression Roadmap
    st.markdown("### 🗓️ Project Weekly Milestones")
    wk1, wk2, wk3, wk4, wk5 = st.columns(5)
    with wk1:
        st.info("**Week 1: Foundations**\n- Problem formulation\n- Dataset exploration\n- Data dictionary & types")
    with wk2:
        st.info("**Week 2: Cleaning**\n- Imputation strategies\n- IQR Outlier filtering\n- Duplicate removal")
    with wk3:
        st.info("**Week 3: EDA & Insights**\n- Correlation heatmaps\n- Class imbalance check\n- Feature distributions")
    with wk4:
        st.info("**Week 4: Pipelines**\n- Leak-free ColumnTransformer\n- Random Forest baseline\n- Model serialization")
    with wk5:
        st.success("**Week 5: Validation & UI**\n- 5-Fold Cross-Validation\n- GridSearchCV tuning\n- Full Interactive Suite")

# ═════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "📊 Exploratory Data Analysis (EDA)":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h1 class="hero-title">📊 Exploratory Data Analysis (EDA)</h1>
            <p class="hero-subtitle">Interactive visual exploration of driver profiles, incident dynamics, and fraud signals</p>
        </div>
    """, unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "🎯 Target & Class Balance",
        "👤 Driver & Demographics",
        "🚗 Incident & Claim Financials",
        "🔥 Correlation & Feature Matrix"
    ])
    
    with eda_tab1:
        st.markdown("#### Target Class Distribution (`fraud_reported`)")
        c1, c2 = st.columns(2)
        
        counts = clean_df['fraud_reported'].value_counts().reset_index()
        counts.columns = ['Status', 'Count']
        counts['Label'] = counts['Status'].map({'N': 'Genuine Claims (N)', 'Y': 'Fraudulent Claims (Y)'})
        
        with c1:
            fig_pie = px.pie(
                counts,
                values='Count',
                names='Label',
                color='Label',
                color_discrete_map={'Genuine Claims (N)': '#2ecc71', 'Fraudulent Claims (Y)': '#e74c3c'},
                hole=0.45,
                title="Claim Status Breakdown"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            fig_bar = px.bar(
                counts,
                x='Label',
                y='Count',
                color='Label',
                color_discrete_map={'Genuine Claims (N)': '#2ecc71', 'Fraudulent Claims (Y)': '#e74c3c'},
                title="Exact Record Counts",
                text='Count'
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.info("💡 **Imbalance Insight**: Fraudulent claims comprise **~25.8%** of all records. To prevent models from defaulting to predicting all genuine cases, `class_weight='balanced'` is utilized across all tree and linear models.")

    with eda_tab2:
        st.markdown("#### Driver Demographics vs Fraud Rate")
        cd1, cd2 = st.columns(2)
        
        with cd1:
            fig_age = px.histogram(
                clean_df,
                x='age_of_driver',
                color='fraud_reported',
                barmode='overlay',
                nbins=30,
                color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'},
                title="Age of Driver Distribution by Claim Outcome"
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
        with cd2:
            fig_inc = px.box(
                clean_df,
                x='fraud_reported',
                y='annual_income',
                color='fraud_reported',
                color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'},
                title="Annual Income ($) vs Fraud Status"
            )
            st.plotly_chart(fig_inc, use_container_width=True)
            
        cd3, cd4 = st.columns(2)
        with cd3:
            edu_df = clean_df.groupby(['high_education', 'fraud_reported']).size().reset_index(name='count')
            edu_df['Education'] = edu_df['high_education'].map({0: 'No Higher Ed', 1: 'Higher Ed Completed'})
            fig_edu = px.bar(
                edu_df,
                x='Education',
                y='count',
                color='fraud_reported',
                barmode='group',
                color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'},
                title="Education Level vs Fraud Reports"
            )
            st.plotly_chart(fig_edu, use_container_width=True)
            
        with cd4:
            safety_fig = px.histogram(
                clean_df,
                x='safety_rating',
                color='fraud_reported',
                marginal="box",
                color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'},
                title="Driver Safety Rating Distribution"
            )
            st.plotly_chart(safety_fig, use_container_width=True)

    with eda_tab3:
        st.markdown("#### Incident Dynamics & Financial Factors")
        ci1, ci2 = st.columns(2)
        
        with ci1:
            fig_claim = px.scatter(
                clean_df,
                x='total_claim',
                y='injury_claim',
                color='fraud_reported',
                color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'},
                opacity=0.6,
                title="Total Claim ($) vs Injury Claim ($)"
            )
            st.plotly_chart(fig_claim, use_container_width=True)
            
        with ci2:
            site_df = clean_df.groupby(['accident_site', 'fraud_reported']).size().reset_index(name='count')
            fig_site = px.bar(
                site_df,
                x='accident_site',
                y='count',
                color='fraud_reported',
                barmode='group',
                color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'},
                title="Claims by Accident Site"
            )
            st.plotly_chart(fig_site, use_container_width=True)
            
        ci3, ci4 = st.columns(2)
        with ci3:
            fig_veh = px.box(
                clean_df,
                x='vehicle_category',
                y='vehicle_price',
                color='fraud_reported',
                color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'},
                title="Vehicle Price by Category & Fraud"
            )
            st.plotly_chart(fig_veh, use_container_width=True)
            
        with ci4:
            fig_days = px.histogram(
                clean_df,
                x='days_open',
                color='fraud_reported',
                barmode='group',
                color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'},
                title="Days Open (Processing Time) vs Outcome"
            )
            st.plotly_chart(fig_days, use_container_width=True)

    with eda_tab4:
        st.markdown("#### Correlation Heatmap & Feature Relationships")
        num_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
        corr = clean_df[num_cols].corr()
        
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Matrix (Numerical Features)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# 3. ALL ALGORITHMS & MODEL COMPARISON (TASK 5)
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "🤖 All Algorithms & Model Comparison":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);">
            <h1 class="hero-title">🤖 All Algorithms & Model Benchmarks</h1>
            <p class="hero-subtitle">Task 5: Comparative Evaluation, 5-Fold Cross-Validation, Overfitting Diagnostics & Hyperparameter Tuning</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🏆 Comprehensive Model Comparison Table (Task 5)")
    st.markdown("Evaluation on 20% stratified hold-out test set (`n=1,528`) with 5-fold cross-validation on training data:")
    
    # Styled dataframe
    def color_status(val):
        if val == 'Good Fit':
            return 'background-color: rgba(46, 204, 113, 0.2); color: #2ecc71; font-weight: bold;'
        elif val == 'Overfitting':
            return 'background-color: rgba(231, 76, 60, 0.2); color: #e74c3c; font-weight: bold;'
        elif val == 'Underfitting':
            return 'background-color: rgba(243, 156, 18, 0.2); color: #f39c12; font-weight: bold;'
        return ''
        
    styled_df = comp_df.style.map(color_status, subset=['Fit Status']).format({
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1-Score': '{:.2%}',
        'ROC-AUC': '{:.2%}',
        'Train Score': '{:.2%}',
        'Test Score': '{:.2%}',
        'CV Mean (5-Fold)': '{:.2%}',
        'CV Std Spread': '{:.4f}'
    })
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual comparison charts
    comp_sub1, comp_sub2, comp_sub3, comp_sub4 = st.tabs([
        "📊 Multi-Metric Comparison",
        "🎯 Overfitting & CV Diagnostics",
        "🔲 Confusion Matrices (All Models)",
        "📈 ROC Curves & Hyperparameter Tuning"
    ])
    
    with comp_sub1:
        st.markdown("#### Metric Benchmark Across All 5 Algorithms")
        
        metrics_melted = pd.melt(
            comp_df,
            id_vars=['Model'],
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            var_name='Metric',
            value_name='Score'
        )
        
        fig_metrics = px.bar(
            metrics_melted,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            text_auto='.1%',
            title="Classification Metrics by Model",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_metrics.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        st.markdown("""
            > **Key Academic Finding:**  
            > In highly imbalanced fraud datasets (~25% fraud cases), naive models like **AdaBoost** achieve high accuracy (74.2%) by predicting class `0` for all instances, resulting in **0.0% Recall** and completely missing fraudulent claims. 
            > In contrast, **Random Forest with balanced weighting** achieves a **Recall of ~55.8%**, capturing the majority of actual fraud cases.
        """)

    with comp_sub2:
        st.markdown("#### Overfitting / Underfitting & Cross-Validation Analysis")
        
        cc1, cc2 = st.columns(2)
        with cc1:
            fit_df = pd.melt(
                comp_df,
                id_vars=['Model'],
                value_vars=['Train Score', 'Test Score'],
                var_name='Set',
                value_name='Accuracy'
            )
            fig_fit = px.bar(
                fit_df,
                x='Model',
                y='Accuracy',
                color='Set',
                barmode='group',
                text_auto='.1%',
                title="Train vs Test Accuracy (Overfitting Check)",
                color_discrete_map={'Train Score': '#3498db', 'Test Score': '#e67e22'}
            )
            fig_fit.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_fit, use_container_width=True)
            
        with cc2:
            fig_cv = px.bar(
                comp_df,
                x='Model',
                y='CV Mean (5-Fold)',
                error_y='CV Std Spread',
                color='Model',
                text_auto='.1%',
                title="5-Fold Cross-Validation Score with Variance Spread (± Std)",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cv.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig_cv, use_container_width=True)

    with comp_sub3:
        st.markdown("#### Confusion Matrix Comparison")
        selected_model_cm = st.selectbox("Select Model to Inspect Confusion Matrix:", list(cm_dict.keys()), key="cm_select")
        
        col_cm1, col_cm2 = st.columns([5, 6])
        cm_matrix = cm_dict[selected_model_cm]
        
        with col_cm1:
            fig_cm = px.imshow(
                cm_matrix,
                text_auto=True,
                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                x=['Genuine (0)', 'Fraud (1)'],
                y=['Genuine (0)', 'Fraud (1)'],
                color_continuous_scale="Blues",
                title=f"Confusion Matrix: {selected_model_cm}"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_cm2:
            tn, fp, fn, tp = cm_matrix.ravel()
            st.markdown(f"##### Detailed Breakdown for `{selected_model_cm}`:")
            st.write(f"• **True Negatives (TN):** `{tn:,}` (Correctly identified genuine claims)")
            st.write(f"• **False Positives (FP):** `{fp:,}` (Genuine claims flagged as fraud for review)")
            st.write(f"• **False Negatives (FN):** `{fn:,}` (Undetected fraudulent claims)")
            st.write(f"• **True Positives (TP):** `{tp:,}` (Successfully intercepted fraud claims)")
            
            # Classification report table
            cr_dict = reports_dict[selected_model_cm]
            cr_df = pd.DataFrame(cr_dict).transpose().round(4)
            st.markdown("##### Detailed Classification Report:")
            st.dataframe(cr_df, use_container_width=True)

    with comp_sub4:
        st.markdown("#### ROC Curves & Hyperparameter Tuning Uplift")
        
        roc_col1, roc_col2 = st.columns(2)
        with roc_col1:
            fig_roc = go.Figure()
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
            
            for idx, (m_name, r_info) in enumerate(roc_dict.items()):
                fig_roc.add_trace(go.Scatter(
                    x=r_info['fpr'],
                    y=r_info['tpr'],
                    mode='lines',
                    name=f"{m_name} (AUC = {r_info['auc']:.2f})",
                    line=dict(color=colors[idx % len(colors)], width=2.5)
                ))
                
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Chance (AUC = 0.50)',
                line=dict(dash='dash', color='gray')
            ))
            
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC) Comparison",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity / Recall)",
                legend=dict(x=0.45, y=0.15)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with roc_col2:
            st.markdown("""
                <div class="custom-card">
                    <div class="custom-card-header">⚙️ GridSearchCV Hyperparameter Tuning (Task 5)</div>
                    <p><b>Model Optimized:</b> Random Forest Classifier</p>
                    <p><b>Optimization Objective:</b> Maximize F1-Score via 5-Fold Cross-Validation</p>
                    <b>Parameter Search Space:</b>
                    <ul>
                        <li><code>n_estimators</code>: [50, 100]</li>
                        <li><code>max_depth</code>: [3, 5, 8]</li>
                        <li><code>min_samples_split</code>: [2, 5]</li>
                    </ul>
                    <hr style="opacity:0.2;">
                    <b>Best Optimal Hyperparameters:</b>
                    <ul>
                        <li><b>Max Depth:</b> <code>3</code> (prevents tree overfitting)</li>
                        <li><b>Min Samples Split:</b> <code>5</code></li>
                        <li><b>Estimators (Trees):</b> <code>100</code></li>
                        <li><b>Class Weight:</b> <code>balanced</code></li>
                    </ul>
                    <p style="color:#2ecc71; font-weight:600;">
                        ✅ Result: Tuned Random Forest achieves balanced fraud detection with Recall = 55.84% and F1 = 36.91%.
                    </p>
                </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 4. INTERACTIVE FRAUD RISK PREDICTOR (LIVE CONSOLE)
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "🛡️ Interactive Fraud Risk Predictor":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">🛡️ Live Fraud Risk Assessment Console</h1>
            <p class="hero-subtitle">Evaluate insurance claims in real-time across any trained machine learning algorithm</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sample pre-fill buttons
    col_preset, col_model_pick = st.columns([6, 6])
    with col_model_pick:
        chosen_algo = st.selectbox(
            "⚡ Active Model for Prediction:",
            list(pipelines.keys()),
            index=0,
            help="Switch between any trained algorithm in real-time to compare predictions!"
        )
    with col_preset:
        st.markdown("**Quick Preset Scenarios:**")
        ps1, ps2, ps3 = st.columns(3)
        load_high_risk = ps1.button("🚨 Load High-Risk Sample", use_container_width=True)
        load_genuine = ps2.button("🛡️ Load Genuine Sample", use_container_width=True)
        reset_form = ps3.button("🔄 Reset Defaults", use_container_width=True)

    # Manage presets in session state
    if 'form_vals' not in st.session_state or reset_form:
        st.session_state.form_vals = {
            'age_of_driver': 43, 'gender': 'Male', 'marital_status': 'Married',
            'annual_income': 37000.0, 'high_education': 'No', 'address_change': 'No',
            'policy_deductible': 500, 'annual_premium': 1000.0, 'zip_code': 50027,
            'property_status': 'Own', 'age_of_vehicle': 5, 'vehicle_category': 'Medium',
            'vehicle_price': 23000.0, 'vehicle_color': 'white', 'claim_date': datetime.today(),
            'accident_site': 'Local', 'witness_present': 'No', 'liab_prct': 50,
            'channel': 'Broker', 'police_report': 'No', 'safety_rating': 76,
            'past_num_of_claims': 1, 'days_open': 8.0, 'form_defects': 4,
            'injury_claim': 1000.0, 'total_claim': 5000.0
        }
    elif load_high_risk:
        st.session_state.form_vals = {
            'age_of_driver': 21, 'gender': 'Male', 'marital_status': 'Single',
            'annual_income': 18000.0, 'high_education': 'No', 'address_change': 'Yes',
            'policy_deductible': 2000, 'annual_premium': 600.0, 'zip_code': 50012,
            'property_status': 'Rent', 'age_of_vehicle': 1, 'vehicle_category': 'Large',
            'vehicle_price': 42000.0, 'vehicle_color': 'red', 'claim_date': datetime.today(),
            'accident_site': 'Highway', 'witness_present': 'No', 'liab_prct': 100,
            'channel': 'Online', 'police_report': 'No', 'safety_rating': 42,
            'past_num_of_claims': 4, 'days_open': 2.0, 'form_defects': 9,
            'injury_claim': 14000.0, 'total_claim': 18000.0
        }
    elif load_genuine:
        st.session_state.form_vals = {
            'age_of_driver': 52, 'gender': 'Female', 'marital_status': 'Married',
            'annual_income': 65000.0, 'high_education': 'Yes', 'address_change': 'No',
            'policy_deductible': 500, 'annual_premium': 1200.0, 'zip_code': 50027,
            'property_status': 'Own', 'age_of_vehicle': 7, 'vehicle_category': 'Compact',
            'vehicle_price': 16000.0, 'vehicle_color': 'silver', 'claim_date': datetime.today(),
            'accident_site': 'Parking Lot', 'witness_present': 'Yes', 'liab_prct': 10,
            'channel': 'Broker', 'police_report': 'Yes', 'safety_rating': 92,
            'past_num_of_claims': 0, 'days_open': 14.0, 'form_defects': 0,
            'injury_claim': 500.0, 'total_claim': 2500.0
        }

    fv = st.session_state.form_vals

    col_input, col_result = st.columns([5, 6], gap="large")
    
    with col_input:
        st.markdown("### 📋 Claim Case Submission Form")
        
        # Driver Profile Card
        with st.container():
            st.markdown("""
                <div class="custom-card">
                    <div class="custom-card-header">👤 Driver Profile</div>
                </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age_of_driver = st.number_input("Age of Driver", min_value=18, max_value=100, value=int(fv['age_of_driver']), step=1)
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if fv['gender'] == 'Male' else 1)
                marital_status = st.selectbox("Marital Status", ["Married", "Single"], index=0 if fv['marital_status'] == 'Married' else 1)
            with c2:
                annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=float(fv['annual_income']), step=1000.0)
                high_education = st.selectbox("Completed High Education?", ["No", "Yes"], index=0 if fv['high_education'] == 'No' else 1)
                address_change = st.selectbox("Address Change (Past 1–5 yrs)?", ["No", "Yes"], index=0 if fv['address_change'] == 'No' else 1)

        # Policy & Vehicle Details Card
        with st.container():
            st.markdown("""
                <div class="custom-card">
                    <div class="custom-card-header">📜 Policy & Vehicle Details</div>
                </div>
            """, unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                policy_deductible = st.selectbox("Policy Deductible ($)", [500, 1000, 2000], index=[500, 1000, 2000].index(fv['policy_deductible']) if fv['policy_deductible'] in [500, 1000, 2000] else 0)
                annual_premium = st.number_input("Annual Premium ($)", min_value=0.0, value=float(fv['annual_premium']), step=50.0)
                zip_code = st.number_input("Zip Code", min_value=0, max_value=99999, value=int(fv['zip_code']), step=1)
                property_status = st.selectbox("Property Status", ["Own", "Rent"], index=0 if fv['property_status'] == 'Own' else 1)
            with c4:
                age_of_vehicle = st.number_input("Age of Vehicle (years)", min_value=0, max_value=50, value=int(fv['age_of_vehicle']), step=1)
                vehicle_category = st.selectbox("Vehicle Category", ["Compact", "Medium", "Large"], index=["Compact", "Medium", "Large"].index(fv['vehicle_category']) if fv['vehicle_category'] in ["Compact", "Medium", "Large"] else 1)
                vehicle_price = st.number_input("Vehicle Price ($)", min_value=0.0, value=float(fv['vehicle_price']), step=1000.0)
                colors_list = ['black', 'blue', 'gray', 'other', 'red', 'silver', 'white']
                vehicle_color = st.selectbox("Vehicle Color", colors_list, index=colors_list.index(fv['vehicle_color']) if fv['vehicle_color'] in colors_list else 6)

        # Incident & Claim Details Card
        with st.container():
            st.markdown("""
                <div class="custom-card">
                    <div class="custom-card-header">🚗 Incident & Claim Details</div>
                </div>
            """, unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                claim_date = st.date_input("Claim Date", value=datetime.today())
                site_options = ["Highway", "Local", "Parking Lot"]
                accident_site = st.selectbox("Accident Site", site_options, index=site_options.index(fv['accident_site']) if fv['accident_site'] in site_options else 1)
                witness_present = st.selectbox("Witness Present?", ["No", "Yes"], index=0 if fv['witness_present'] == 'No' else 1)
                liab_prct = st.slider("Liability Percentage (%)", 0, 100, int(fv['liab_prct']))
                channel_opts = ["Broker", "Online", "Phone"]
                channel = st.selectbox("Agent Channel", channel_opts, index=channel_opts.index(fv['channel']) if fv['channel'] in channel_opts else 0)
            with c6:
                police_report = st.selectbox("Police Report Filed?", ["No", "Yes"], index=0 if fv['police_report'] == 'No' else 1)
                safety_rating = st.slider("Driver Safety Rating", 0, 100, int(fv['safety_rating']))
                past_num_of_claims = st.number_input("Past Number of Claims", min_value=0, max_value=10, value=int(fv['past_num_of_claims']), step=1)
                days_open = st.number_input("Days Open (Processing)", min_value=0.0, value=float(fv['days_open']), step=1.0)
                form_defects = st.number_input("Form Defects Count", min_value=0, max_value=20, value=int(fv['form_defects']), step=1)
                injury_claim = st.number_input("Injury Claim Amount ($)", min_value=0.0, value=float(fv['injury_claim']), step=100.0)
                total_claim = st.number_input("Total Claim Amount ($)", min_value=0.0, value=float(fv['total_claim']), step=500.0)

        inputs_dict = {
            'age_of_driver': age_of_driver, 'gender': gender, 'marital_status': marital_status,
            'annual_income': annual_income, 'high_education': high_education, 'address_change': address_change,
            'policy_deductible': policy_deductible, 'annual_premium': annual_premium, 'zip_code': zip_code,
            'property_status': property_status, 'age_of_vehicle': age_of_vehicle, 'vehicle_category': vehicle_category,
            'vehicle_price': vehicle_price, 'vehicle_color': vehicle_color, 'claim_date': claim_date,
            'accident_site': accident_site, 'witness_present': witness_present, 'liab_prct': liab_prct,
            'channel': channel, 'police_report': police_report, 'safety_rating': safety_rating,
            'past_num_of_claims': past_num_of_claims, 'days_open': days_open, 'form_defects': form_defects,
            'injury_claim': injury_claim, 'total_claim': total_claim
        }

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🛡️ Execute Fraud Risk Assessment", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### 🔍 Risk Assessment Verdict & Insights")
        
        # Run prediction
        active_pipeline = pipelines[chosen_algo]
        
        if total_claim < injury_claim:
            st.error("⚠️ **Validation Error**: Total Claim Amount ($) must be greater than or equal to Injury Claim Amount ($).")
        else:
            input_df = ml_engine.format_single_input(inputs_dict)
            pred = active_pipeline.predict(input_df)[0]
            
            if hasattr(active_pipeline, "predict_proba"):
                probs = active_pipeline.predict_proba(input_df)[0]
                fraud_prob = probs[1]
                genuine_prob = probs[0]
            else:
                fraud_prob = 1.0 if pred == 1 else 0.0
                genuine_prob = 1.0 - fraud_prob
                
            fraud_pct = fraud_prob * 100
            genuine_pct = genuine_prob * 100
            
            # 1. Verdict Box
            if pred == 1:
                st.markdown(f"""
                    <div class="verdict-box verdict-high">
                        <div class="verdict-badge high">🚨 HIGH FRAUD RISK FLAGGED</div>
                        <div class="score-number high">{fraud_pct:.1f}%</div>
                        <p style="font-size:1.1rem; margin:0.5rem 0;">Estimated Fraud Risk Probability</p>
                        <p style="opacity:0.85; line-height:1.5;">
                            This claim exhibits strong structural characteristics consistent with historical fraudulent submissions.
                        </p>
                        <div class="action-badge" style="color:#e74c3c;">
                            <b>Recommended Action:</b> Escalate to Special Investigation Unit (SIU) for comprehensive audit.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="verdict-box verdict-low">
                        <div class="verdict-badge low">🛡️ LOW FRAUD RISK (GENUINE)</div>
                        <div class="score-number low">{genuine_pct:.1f}%</div>
                        <p style="font-size:1.1rem; margin:0.5rem 0;">Estimated Genuine Claim Confidence</p>
                        <p style="opacity:0.85; line-height:1.5;">
                            The claim profile conforms to standard, non-fraudulent submission patterns.
                        </p>
                        <div class="action-badge" style="color:#2ecc71;">
                            <b>Recommended Action:</b> Approve for standard settlement disbursement pipeline.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            # 2. Probability Gauge / Breakdown
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Fraud Risk Score Meter ({chosen_algo})", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                    'bar': {'color': "#e74c3c" if pred == 1 else "#2ecc71"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.2)"},
                        {'range': [30, 60], 'color': "rgba(241, 196, 15, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(231, 76, 60, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # 3. Model Factor Importance (if available)
            try:
                clf = active_pipeline.named_steps['classifier']
                prep = active_pipeline.named_steps['preprocessor']
                if hasattr(clf, 'feature_importances_') and hasattr(prep, 'get_feature_names_out'):
                    feat_names = [f.split('__')[-1].replace('_', ' ').title() for f in prep.get_feature_names_out()]
                    imp = clf.feature_importances_
                    imp_df = pd.DataFrame({'Factor': feat_names, 'Importance': imp}).sort_values('Importance', ascending=True).tail(7)
                    
                    fig_imp = px.bar(
                        imp_df,
                        x='Importance',
                        y='Factor',
                        orientation='h',
                        title="Top Influencing Features for this Model",
                        color='Importance',
                        color_continuous_scale="Viridis"
                    )
                    fig_imp.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_imp, use_container_width=True)
            except Exception:
                pass

            # 4. Summary of Case
            with st.expander("📋 Submitted Case Snapshot", expanded=True):
                cs1, cs2 = st.columns(2)
                with cs1:
                    st.write(f"**Driver:** {age_of_driver} yrs ({gender}, {marital_status})")
                    st.write(f"**Income:** ${annual_income:,.2f}")
                    st.write(f"**Vehicle:** {age_of_vehicle} yrs ({vehicle_category}, ${vehicle_price:,.2f})")
                    st.write(f"**Site:** {accident_site}")
                with cs2:
                    st.write(f"**Total Claim:** ${total_claim:,.2f}")
                    st.write(f"**Injury Claim:** ${injury_claim:,.2f}")
                    st.write(f"**Past Claims:** {past_num_of_claims}")
                    st.write(f"**Algorithm:** `{chosen_algo}`")

# ═════════════════════════════════════════════════════════════════════════════
# 5. BATCH CLAIMS PREDICTOR (CSV UPLOAD)
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "📂 Batch Claims Predictor (CSV)":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);">
            <h1 class="hero-title">📂 Batch Claims Audit & CSV Processor</h1>
            <p class="hero-subtitle">Upload claims files or process instant batches to screen hundreds of claims in seconds</p>
        </div>
    """, unsafe_allow_html=True)
    
    b_col1, b_col2 = st.columns([7, 5])
    with b_col1:
        batch_model_name = st.selectbox("Select Model for Batch Inference:", list(pipelines.keys()), index=0)
    with b_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample_batch = st.button("🧪 Load Sample Batch (15 Historical Claims)", use_container_width=True)
        
    uploaded_file = st.file_uploader("Upload Claims CSV File", type=["csv"])
    
    df_to_process = None
    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV with {df_to_process.shape[0]} rows and {df_to_process.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif load_sample_batch:
        # Sample random rows from cleaned data
        df_to_process = clean_df.drop(columns=['fraud_reported'], errors='ignore').sample(15, random_state=42)
        st.info("Loaded 15 sample claims from dataset.")
        
    if df_to_process is not None:
        st.markdown("### 🔍 Processed Batch Results")
        
        # Prepare pipeline
        b_pipe = pipelines[batch_model_name]
        
        try:
            batch_preds = b_pipe.predict(df_to_process)
            if hasattr(b_pipe, "predict_proba"):
                batch_probs = b_pipe.predict_proba(df_to_process)[:, 1]
            else:
                batch_probs = [1.0 if p == 1 else 0.0 for p in batch_preds]
                
            results_batch = df_to_process.copy()
            results_batch['Predicted_Verdict'] = ['🚨 HIGH RISK' if p == 1 else '🛡️ GENUINE' for p in batch_preds]
            results_batch['Fraud_Probability_%'] = (np.array(batch_probs) * 100).round(1)
            
            # Summary stats
            total_b = len(batch_preds)
            fraud_b = sum(batch_preds)
            gen_b = total_b - fraud_b
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Batch Claims", f"{total_b:,}")
            k2.metric("Flagged High Risk", f"{fraud_b:,}", f"{(fraud_b/total_b)*100:.1f}%")
            k3.metric("Passed Genuine", f"{gen_b:,}", f"{(gen_b/total_b)*100:.1f}%")
            
            # Display results
            st.dataframe(results_batch, use_container_width=True)
            
            # Download CSV
            csv_data = results_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Annotated Predictions CSV",
                data=csv_data,
                file_name=f"fraud_predictions_{batch_model_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error executing batch inference: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. SEMESTER 5 PROJECT DOCUMENTATION
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "📑 Semester 5 Project Documentation":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #2C3E50 0%, #4CA1AF 100%);">
            <h1 class="hero-title">📑 Academic Documentation & SOP Report</h1>
            <p class="hero-subtitle">Machine Learning Project Portfolio — Complete Task Breakdown & Theoretical Basis</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📌 Week 1: Problem Definition & Data Understanding", expanded=True):
        st.markdown("""
        - **Objective:** Supervised binary classification to determine vehicle insurance claim authenticity.
        - **Target Attribute:** `fraud_reported` (`Y` = Fraud / 1, `N` = Genuine / 0).
        - **Dataset Scale:** 12,002 initial claim records across 29 continuous and categorical features.
        """)
        
    with st.expander("📌 Week 2: Data Cleaning & Preprocessing", expanded=False):
        st.markdown("""
        - **Data Quality Remediations:** Identified corrupted placeholder values (`*`) in numeric fields and coerced to NaN.
        - **Imputation:** Median imputation applied to skewed numeric attributes (`injury_claim`, `age_of_vehicle`, `marital_status`, `witness_present`); Mode imputation applied to dates and categorical fields.
        - **Outlier Mitigation:** Interquartile Range (IQR) filter applied to continuous numerical attributes to remove noisy recording artifacts.
        """)

    with st.expander("📌 Week 3: Exploratory Data Analysis (EDA)", expanded=False):
        st.markdown("""
        - **Class Imbalance Discovered:** 25.8% fraud vs 74.2% genuine claims.
        - **Risk Signals:** Higher total claim amounts relative to policy premium and younger driver age groups exhibit statistically higher fraud probabilities.
        """)

    with st.expander("📌 Week 4: Pipelines & Leak-Free Architecture", expanded=False):
        st.markdown("""
        - **Pipeline Encapsulation:** Scikit-learn `ColumnTransformer` with `StandardScaler` and `OrdinalEncoder`.
        - **Anti-Data-Leakage:** Transformer parameters fit strictly on training partitions only (`random_state=42`, stratified split).
        """)

    with st.expander("📌 Week 5: Multi-Algorithm Benchmarks & Tuning (Task 5)", expanded=True):
        st.markdown("""
        - **Evaluated Algorithms:** Logistic Regression, Decision Tree, Random Forest (Bagging), AdaBoost, Tuned Random Forest.
        - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, 5-Fold Cross-Validation mean & spread.
        - **Overfitting Diagnosis:** Monitored Train Score vs Test Score gap to guarantee generalization.
        - **GridSearchCV Hyperparameter Optimization:** Selected 100 estimators, max depth 3, min samples split 5, class_weight balanced.
        """)
