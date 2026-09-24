import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
from datetime import datetime, date

warnings.filterwarnings('ignore')

# ── Import ML Engine ────────────────────────────────────────────────────────
import ml_engine

# ── Streamlit Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI • Enterprise Insurance Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Modern UI & Custom Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains+Mono', monospace !important;
    }

    /* Hero Banners */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(0, 210, 255, 0.2) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* KPI Metrics Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 1.1rem;
        margin-bottom: 1.8rem;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 1.25rem 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: #00d2ff;
        box-shadow: 0 10px 25px rgba(0, 210, 255, 0.15);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #00d2ff;
        margin-top: 0.3rem;
        letter-spacing: -0.5px;
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.75;
        font-weight: 600;
    }
    .kpi-subtitle {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 0.2rem;
    }

    /* Content Cards */
    .custom-card {
        background: rgba(128, 128, 128, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    .custom-card-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* Verdict Boxes */
    .verdict-box {
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        animation: fadeIn 0.4s ease-in-out;
        position: relative;
    }
    .verdict-high {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15) 0%, rgba(192, 57, 43, 0.05) 100%);
        border: 2px solid #e74c3c;
    }
    .verdict-low {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15) 0%, rgba(39, 174, 96, 0.05) 100%);
        border: 2px solid #2ecc71;
    }
    .verdict-badge {
        display: inline-block;
        padding: 0.5rem 1.8rem;
        font-weight: 800;
        font-size: 1.15rem;
        border-radius: 50px;
        letter-spacing: 0.06em;
        color: white;
        margin-bottom: 1.2rem;
    }
    .verdict-badge.high { background: linear-gradient(90deg, #e74c3c, #c0392b); }
    .verdict-badge.low { background: linear-gradient(90deg, #2ecc71, #27ae60); }

    .score-number {
        font-size: 3.8rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.6rem 0;
        letter-spacing: -1px;
    }
    .score-number.high { color: #e74c3c; }
    .score-number.low { color: #2ecc71; }

    .action-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.6rem 1.2rem;
        background: rgba(128, 128, 128, 0.1);
        border-radius: 10px;
        font-size: 0.92rem;
        font-weight: 500;
    }

    /* Model Vote Card */
    .model-vote-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .model-vote-card:hover {
        transform: translateY(-2px);
    }
    
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .status-pill.fraud { background: rgba(231, 76, 60, 0.2); color: #e74c3c; }
    .status-pill.genuine { background: rgba(46, 204, 113, 0.2); color: #2ecc71; }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Big Sidebar Navigation Buttons */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        text-align: left !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        padding: 0.85rem 1.15rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        margin-bottom: 0.5rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateX(5px) scale(1.01) !important;
        border-color: #00d2ff !important;
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.25) !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%) !important;
        color: white !important;
        border: 1.5px solid #00d2ff !important;
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.35) !important;
        transform: translateX(4px) !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.04) !important;
        color: inherit !important;
    }

    /* Top Quick Switcher */
    .top-nav-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# ── Load and Cache ML Pipeline and Data ─────────────────────────────────────
@st.cache_resource(show_spinner="Training and benchmarking all 5 machine learning algorithms...")
def get_ml_suite():
    return ml_engine.train_all_models()

@st.cache_data(show_spinner="Ingesting and preprocessing dataset...")
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

# ── Session State for Navigation ─────────────────────────────────────────────
if 'active_page' not in st.session_state:
    st.session_state.active_page = "🏠 1. Executive Dashboard & Pipeline"

nav_pages = [
    "🏠 1. Executive Dashboard & Pipeline",
    "📊 2. Exploratory Data Analysis (EDA)",
    "🤖 3. Model Benchmark & Task 5 Lab",
    "🛡️ 4. Live Risk Predictor & Consensus",
    "🔮 5. What-If Sensitivity Simulator",
    "💰 6. Financial Impact & ROI Calculator",
    "📂 7. Batch Claims Audit (CSV)",
    "📑 8. Project SOP & Documentation"
]

# ── Sidebar Navigation (Big Button-Based UI) ─────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 0.4rem 0;">
            <div style="font-size: 3.2rem; filter: drop-shadow(0 4px 10px rgba(0,210,255,0.4));">🛡️</div>
            <h2 style="margin: 0.3rem 0 0 0; font-weight: 900; font-size: 1.45rem; letter-spacing: -0.5px;">FraudShield AI</h2>
            <p style="font-size: 0.82rem; opacity: 0.75; margin-top: 0.2rem;">Enterprise Claims Risk Intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.6; margin: 0.8rem 0 0.4rem 0;'>Navigation Menu</p>", unsafe_allow_html=True)
    
    # Render Big Modern Navigation Buttons
    for page_name in nav_pages:
        is_active = (st.session_state.active_page == page_name)
        btn_type = "primary" if is_active else "secondary"
        if st.button(page_name, key=f"btn_nav_{page_name}", type=btn_type, use_container_width=True):
            st.session_state.active_page = page_name
            st.rerun()
            
    st.markdown("---")
    st.markdown("### ⚡ Live System Telemetry")
    st.markdown(f"• **Clean Records:** `{clean_df.shape[0]:,}`")
    st.markdown(f"• **Features Encoded:** `{len(ml_data['feature_cols'])}`")
    st.markdown("• **Best Model:** `Random Forest (Tuned)`")
    st.markdown(f"• **Peak Test ROC-AUC:** `71.74%`")
    st.markdown(f"• **Engine Status:** `🟢 Active & Ready`")
    
    st.markdown("---")
    st.caption("Machine Learning Sem-5 Project • Vehicle Insurance Fraud Analytics")

app_mode = st.session_state.active_page

# ── Top Bar Quick Jump Button Bar ───────────────────────────────────────────
with st.container():
    st.markdown("<div style='margin-bottom: 0.8rem;'>", unsafe_allow_html=True)
    t_c1, t_c2, t_c3, t_c4, t_c5, t_c6, t_c7, t_c8 = st.columns(8)
    top_cols = [t_c1, t_c2, t_c3, t_c4, t_c5, t_c6, t_c7, t_c8]
    short_titles = [
        "🏠 Dashboard", "📊 EDA Lab", "🤖 Models", "🛡️ Predictor",
        "🔮 What-If", "💰 ROI Math", "📂 Batch CSV", "📑 SOP Docs"
    ]
    for idx, (p_name, s_title) in enumerate(zip(nav_pages, short_titles)):
        is_sel = (st.session_state.active_page == p_name)
        b_type = "primary" if is_sel else "secondary"
        with top_cols[idx]:
            if st.button(s_title, key=f"top_nav_{idx}", type=b_type, use_container_width=True):
                st.session_state.active_page = p_name
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. EXECUTIVE DASHBOARD & PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
if app_mode == "🏠 1. Executive Dashboard & Pipeline":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">🛡️ Insurance Fraud Intelligence Platform</h1>
            <p class="hero-subtitle">Automated multi-algorithm supervised learning system engineered to detect, classify, and mitigate auto insurance fraud before disbursement.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # KPI Grid
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Raw Claims Ingested</div>
                <div class="kpi-value">{len(raw_df):,}</div>
                <div class="kpi-subtitle">Initial raw records</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Cleaned Dataset</div>
                <div class="kpi-value">{len(clean_df):,}</div>
                <div class="kpi-subtitle">Post-IQR & Imputation</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Fraud Prevalence</div>
                <div class="kpi-value" style="color:#e74c3c;">25.79%</div>
                <div class="kpi-subtitle">Minority class ratio</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Algorithms Benchmarked</div>
                <div class="kpi-value" style="color:#2ecc71;">5 Models</div>
                <div class="kpi-subtitle">Linear, Tree & Ensembles</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Peak Fraud Recall</div>
                <div class="kpi-value" style="color:#f39c12;">55.84%</div>
                <div class="kpi-subtitle">Tuned Random Forest</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([6, 5])
    
    with c_left:
        st.markdown("""
            <div class="custom-card">
                <div class="custom-card-header">🎯 Project Scope & Architecture Highlights</div>
                <p style="line-height: 1.7; font-size: 0.95rem;">
                    Auto insurance fraud drains billions of dollars annually from insurance providers, driving premium costs higher for legitimate customers.
                    This production-grade platform deploys <b>leak-free scikit-learn pipelines</b>, handling missing values, placeholder encodings, 
                    outlier bounds, and imbalanced target distributions.
                </p>
                <div style="margin-top: 1rem;">
                    <b>Core Engineering Pillars:</b>
                    <ul style="line-height: 1.8; margin-top: 0.4rem; font-size:0.92rem;">
                        <li><b>Imbalance Handling:</b> Cost-sensitive balanced class weighting prevents trivial majority guessing.</li>
                        <li><b>Data Hygiene:</b> Automated numeric coercion of corrupted placeholders (<code>*</code>), median/mode imputation.</li>
                        <li><b>Comparative Benchmarking:</b> 5-fold cross-validation with variance spread analysis across 5 algorithms.</li>
                        <li><b>Hyperparameter Search:</b> Exhaustive <code>GridSearchCV</code> tuning for depth, tree counts, and split thresholds.</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with c_right:
        st.markdown("""
            <div class="custom-card">
                <div class="custom-card-header">🔄 End-to-End Pipeline Workflow</div>
                <div style="display:flex; flex-direction:column; gap:0.75rem; font-size:0.9rem;">
                    <div style="padding:0.7rem; background:rgba(0,188,212,0.08); border-left:4px solid #00bcd4; border-radius:6px;">
                        <b>1. Ingestion & Sanitization:</b> Parse raw CSV, coerce <code>*</code> placeholders to NaN, apply IQR continuous outlier filtering.
                    </div>
                    <div style="padding:0.7rem; background:rgba(52,152,219,0.08); border-left:4px solid #3498db; border-radius:6px;">
                        <b>2. Preprocessing & Encoding:</b> <code>StandardScaler</code> for numeric features + <code>OrdinalEncoder</code> for categories.
                    </div>
                    <div style="padding:0.7rem; background:rgba(155,89,182,0.08); border-left:4px solid #9b59b6; border-radius:6px;">
                        <b>3. Modeling & Optimization:</b> Logistic Regression, Decision Tree, Random Forest, AdaBoost & Tuned Random Forest.
                    </div>
                    <div style="padding:0.7rem; background:rgba(46,204,113,0.08); border-left:4px solid #2ecc71; border-radius:6px;">
                        <b>4. Real-time SIU Decision Support:</b> Live risk scoring gauge, 5-model consensus voting, and batch claims audit.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Interactive Dataset Explorer Card
    with st.expander("🔍 Interactive Cleaned Dataset Explorer & Column Dictionary", expanded=False):
        d_col1, d_col2 = st.columns([4, 2])
        with d_col1:
            filter_cat = st.multiselect("Filter by Vehicle Category:", clean_df['vehicle_category'].unique().tolist(), default=clean_df['vehicle_category'].unique().tolist())
        with d_col2:
            filter_fraud = st.selectbox("Filter Fraud Status:", ["All Claims", "Fraudulent Claims Only (Y)", "Genuine Claims Only (N)"])
            
        filtered_view = clean_df[clean_df['vehicle_category'].isin(filter_cat)]
        if filter_fraud == "Fraudulent Claims Only (Y)":
            filtered_view = filtered_view[filtered_view['fraud_reported'] == 'Y']
        elif filter_fraud == "Genuine Claims Only (N)":
            filtered_view = filtered_view[filtered_view['fraud_reported'] == 'N']
            
        st.dataframe(filtered_view.head(50), use_container_width=True)
        st.caption(f"Showing top 50 records of {len(filtered_view):,} matching claims.")

    # Weekly Milestone Roadmap
    st.markdown("### 🗓️ Project Milestone Roadmap")
    wk1, wk2, wk3, wk4, wk5 = st.columns(5)
    with wk1:
        st.info("**Week 1: Foundations**\n- Problem formulation\n- Raw data profiling\n- Target definition")
    with wk2:
        st.info("**Week 2: Data Cleaning**\n- `*` placeholder fix\n- Median/mode impute\n- IQR outlier removal")
    with wk3:
        st.info("**Week 3: EDA & Signals**\n- Correlation heatmaps\n- Class imbalance check\n- Demographic analysis")
    with wk4:
        st.info("**Week 4: Pipelines**\n- Leak-free ColumnTransformer\n- Baseline modeling\n- Serialization")
    with wk5:
        st.success("**Week 5: Validation & UI**\n- 5-Fold Cross-Validation\n- GridSearchCV tuning\n- Full Interactive Suite")

# ═════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "📊 2. Exploratory Data Analysis (EDA)":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h1 class="hero-title">📊 Exploratory Data Analysis (EDA)</h1>
            <p class="hero-subtitle">Interactive visual analytics of driver profiles, incident dynamics, financial distributions, and fraud indicators.</p>
        </div>
    """, unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "🎯 Target & Imbalance",
        "👤 Driver Demographics",
        "🚗 Incident & Financials",
        "🔥 Correlation Heatmap",
        "🧪 Custom Feature Explorer"
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
                hole=0.48,
                title="Claim Outcome Proportion"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1.5)))
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
            
        st.info("💡 **Imbalance Insight**: Fraudulent claims comprise **~25.8%** of all records. Cost-sensitive learning (`class_weight='balanced'`) is essential to prevent models from predicting 100% genuine claims.")

    with eda_tab2:
        st.markdown("#### Driver Demographics vs Fraud Outcome")
        cd1, cd2 = st.columns(2)
        
        with cd1:
            fig_age = px.histogram(
                clean_df,
                x='age_of_driver',
                color='fraud_reported',
                barmode='overlay',
                nbins=30,
                color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'},
                title="Age of Driver Distribution by Fraud Outcome"
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
        st.markdown("#### Correlation Heatmap & Feature Interrelationships")
        num_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
        corr = clean_df[num_cols].corr()
        
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Matrix (Continuous Features)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with eda_tab5:
        st.markdown("#### Dynamic Feature Relationship Inspector")
        fx1, fx2, fx3 = st.columns(3)
        with fx1:
            sel_x = st.selectbox("Select X-Axis Feature:", num_cols, index=num_cols.index('total_claim') if 'total_claim' in num_cols else 0)
        with fx2:
            sel_y = st.selectbox("Select Y-Axis Feature:", num_cols, index=num_cols.index('annual_income') if 'annual_income' in num_cols else 1)
        with fx3:
            chart_type = st.selectbox("Chart Type:", ["Scatter Plot", "2D Density Contour", "Box Plot"])
            
        if chart_type == "Scatter Plot":
            fig_custom = px.scatter(clean_df, x=sel_x, y=sel_y, color='fraud_reported', color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'}, opacity=0.65, title=f"{sel_x} vs {sel_y}")
        elif chart_type == "2D Density Contour":
            fig_custom = px.density_contour(clean_df, x=sel_x, y=sel_y, color='fraud_reported', color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'}, title=f"Density Contour: {sel_x} vs {sel_y}")
        else:
            fig_custom = px.box(clean_df, x='fraud_reported', y=sel_y, color='fraud_reported', color_discrete_map={'N': '#2ecc71', 'Y': '#e74c3c'}, title=f"{sel_y} by Fraud Outcome")
        st.plotly_chart(fig_custom, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# 3. ALL ALGORITHMS & MODEL BENCHMARK (TASK 5)
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "🤖 3. Model Benchmark & Task 5 Lab":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);">
            <h1 class="hero-title">🤖 Multi-Algorithm Benchmark & Task 5 Lab</h1>
            <p class="hero-subtitle">Comprehensive evaluation, 5-Fold Cross-Validation, Overfitting Diagnostics, and GridSearchCV Hyperparameter Optimization.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🏆 Comprehensive Model Performance Benchmark")
    st.markdown("Hold-out stratified test evaluation (`n=1,528`, 20%) alongside 5-fold cross-validation on training data:")
    
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
    
    comp_sub1, comp_sub2, comp_sub3, comp_sub4 = st.tabs([
        "📊 Multi-Metric Comparison",
        "🎯 Overfitting & CV Diagnostics",
        "🔲 Confusion Matrices (All Models)",
        "📈 ROC Curves & GridSearchCV Details"
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
            > In imbalanced fraud datasets (~25% fraud), naive boosting (AdaBoost) achieves high nominal accuracy (74.2%) by classifying nearly all instances as Genuine (`0`), resulting in **0.0% Recall** and failing to detect fraud. 
            > In contrast, **Random Forest with balanced class weighting** achieves **Recall ~55.8%**, successfully intercepting the majority of fraudulent claims.
        """)

    with comp_sub2:
        st.markdown("#### Generalization & 5-Fold Cross-Validation Analysis")
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
                title="Train Score vs Test Score (Overfitting Gap)",
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
        st.markdown("#### Confusion Matrix Inspector")
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
            st.write(f"• **True Negatives (TN):** `{tn:,}` (Correct genuine approvals)")
            st.write(f"• **False Positives (FP):** `{fp:,}` (Genuine flagged for SIU audit)")
            st.write(f"• **False Negatives (FN):** `{fn:,}` (Undetected fraudulent claims)")
            st.write(f"• **True Positives (TP):** `{tp:,}` (Successfully caught fraud claims)")
            
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
                title="Receiver Operating Characteristic (ROC) Curves",
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
                    <p><b>Objective Function:</b> Maximize F1-Score via Stratified 5-Fold CV</p>
                    <b>Parameter Search Space:</b>
                    <ul>
                        <li><code>n_estimators</code>: [50, 100]</li>
                        <li><code>max_depth</code>: [3, 5, 8]</li>
                        <li><code>min_samples_split</code>: [2, 5]</li>
                    </ul>
                    <hr style="opacity:0.2;">
                    <b>Best Optimal Hyperparameters:</b>
                    <ul>
                        <li><b>Max Depth:</b> <code>3</code> (regularizes trees & stops overfitting)</li>
                        <li><b>Min Samples Split:</b> <code>5</code></li>
                        <li><b>Estimators (Trees):</b> <code>100</code></li>
                        <li><b>Class Weight:</b> <code>balanced</code></li>
                    </ul>
                    <p style="color:#2ecc71; font-weight:600; margin-top:0.5rem;">
                        ✅ Result: Tuned Random Forest achieves balanced fraud detection with Recall = 55.84% and F1 = 36.91%.
                    </p>
                </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 4. INTERACTIVE FRAUD RISK PREDICTOR & CONSENSUS
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "🛡️ 4. Live Risk Predictor & Consensus":
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">🛡️ Live Fraud Risk Assessment Console</h1>
            <p class="hero-subtitle">Evaluate insurance claims in real-time, test scenario presets, and inspect consensus across all 5 trained models.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Preset scenarios
    col_preset, col_model_pick = st.columns([6, 6])
    with col_model_pick:
        chosen_algo = st.selectbox(
            "⚡ Primary Model for Assessment:",
            list(pipelines.keys()),
            index=0,
            help="Switch between any trained algorithm in real-time!"
        )
    with col_preset:
        st.markdown("**Quick Preset Case Scenarios:**")
        ps1, ps2, ps3 = st.columns(3)
        load_high_risk = ps1.button("🚨 High-Risk Suspect", use_container_width=True)
        load_genuine = ps2.button("🛡️ Genuine Driver", use_container_width=True)
        reset_form = ps3.button("🔄 Reset Defaults", use_container_width=True)

    if 'form_vals' not in st.session_state or reset_form:
        st.session_state.form_vals = {
            'age_of_driver': 43, 'gender': 'Male', 'marital_status': 'Married',
            'annual_income': 37000.0, 'high_education': 'No', 'address_change': 'No',
            'policy_deductible': 500, 'annual_premium': 1000.0, 'zip_code': 50027,
            'property_status': 'Own', 'age_of_vehicle': 5, 'vehicle_category': 'Medium',
            'vehicle_price': 23000.0, 'vehicle_color': 'white', 'claim_date': date.today(),
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
            'vehicle_price': 42000.0, 'vehicle_color': 'red', 'claim_date': date.today(),
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
            'vehicle_price': 16000.0, 'vehicle_color': 'silver', 'claim_date': date.today(),
            'accident_site': 'Parking Lot', 'witness_present': 'Yes', 'liab_prct': 10,
            'channel': 'Broker', 'police_report': 'Yes', 'safety_rating': 92,
            'past_num_of_claims': 0, 'days_open': 14.0, 'form_defects': 0,
            'injury_claim': 500.0, 'total_claim': 2500.0
        }

    fv = st.session_state.form_vals
    col_input, col_result = st.columns([5, 6], gap="large")
    
    with col_input:
        st.markdown("### 📋 Claim Case Submission Form")
        
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

        with st.container():
            st.markdown("""
                <div class="custom-card">
                    <div class="custom-card-header">🚗 Incident & Claim Details</div>
                </div>
            """, unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                claim_date = st.date_input("Claim Date", value=fv.get('claim_date', date.today()))
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

    with col_result:
        st.markdown("### 🔍 Risk Assessment Verdict & Insights")
        
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
            
            if pred == 1:
                st.markdown(f"""
                    <div class="verdict-box verdict-high">
                        <div class="verdict-badge high">🚨 HIGH FRAUD RISK FLAGGED</div>
                        <div class="score-number high">{fraud_pct:.1f}%</div>
                        <p style="font-size:1.1rem; margin:0.4rem 0;">Estimated Fraud Risk Probability</p>
                        <p style="opacity:0.85; line-height:1.5; font-size:0.95rem;">
                            This claim exhibits structural anomaly indicators consistent with known historical fraud patterns.
                        </p>
                        <div class="action-badge" style="color:#e74c3c;">
                            <b>Recommended SIU Action:</b> Freeze payout disbursement and escalate for Special Investigation audit.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="verdict-box verdict-low">
                        <div class="verdict-badge low">🛡️ LOW RISK — GENUINE CLAIM</div>
                        <div class="score-number low">{genuine_pct:.1f}%</div>
                        <p style="font-size:1.1rem; margin:0.4rem 0;">Estimated Genuine Claim Confidence</p>
                        <p style="opacity:0.85; line-height:1.5; font-size:0.95rem;">
                            The claim profile conforms to standard authentic submission dynamics.
                        </p>
                        <div class="action-badge" style="color:#2ecc71;">
                            <b>Recommended SIU Action:</b> Fast-track approval for standard settlement disbursement.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Fraud Risk Meter ({chosen_algo})", 'font': {'size': 18}},
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

            # ⚡ All 5 Models Simultaneous Consensus
            st.markdown("#### ⚡ 5-Model Consensus Voting Matrix")
            all_votes = []
            cols_v = st.columns(5)
            for idx, (m_name, p_pipe) in enumerate(pipelines.items()):
                m_pred = p_pipe.predict(input_df)[0]
                if hasattr(p_pipe, "predict_proba"):
                    m_prob = p_pipe.predict_proba(input_df)[0][1] * 100
                else:
                    m_prob = 100.0 if m_pred == 1 else 0.0
                all_votes.append(m_pred)
                
                short_name = m_name.replace(' (GridSearchCV)', '').replace(' (Bagging)', '')
                with cols_v[idx]:
                    if m_pred == 1:
                        st.markdown(f"""
                            <div class="model-vote-card" style="border-top: 3px solid #e74c3c;">
                                <div style="font-size:0.75rem; font-weight:700; height:32px;">{short_name}</div>
                                <div class="status-pill fraud" style="margin: 0.3rem 0;">🚨 FRAUD</div>
                                <div style="font-size:0.85rem; font-weight:700; color:#e74c3c;">{m_prob:.1f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="model-vote-card" style="border-top: 3px solid #2ecc71;">
                                <div style="font-size:0.75rem; font-weight:700; height:32px;">{short_name}</div>
                                <div class="status-pill genuine" style="margin: 0.3rem 0;">🛡️ GENUINE</div>
                                <div style="font-size:0.85rem; font-weight:700; color:#2ecc71;">{m_prob:.1f}%</div>
                            </div>
                        """, unsafe_allow_html=True)

            fraud_votes = sum(all_votes)
            st.caption(f"Consensus: **{fraud_votes} of 5 models** flagged this claim as Fraudulent.")

            # Top Influencing Features
            try:
                clf = active_pipeline.named_steps['classifier']
                prep = active_pipeline.named_steps['preprocessor']
                if hasattr(clf, 'feature_importances_') and hasattr(prep, 'get_feature_names_out'):
                    feat_names = [f.split('__')[-1].replace('_', ' ').title() for f in prep.get_feature_names_out()]
                    imp = clf.feature_importances_
                    imp_df = pd.DataFrame({'Factor': feat_names, 'Importance': imp}).sort_values('Importance', ascending=True).tail(6)
                    
                    fig_imp = px.bar(
                        imp_df,
                        x='Importance',
                        y='Factor',
                        orientation='h',
                        title="Top Influencing Risk Factors (Model Feature Importance)",
                        color='Importance',
                        color_continuous_scale="Viridis"
                    )
                    fig_imp.update_layout(height=240, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_imp, use_container_width=True)
            except Exception:
                pass

            # Download Investigation Dossier
            dossier_text = f"""=======================================================
FRAUDSHIELD AI • SPECIAL INVESTIGATION UNIT (SIU) DOSSIER
Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Primary Model: {chosen_algo}
=======================================================

1. RISK ASSESSMENT VERDICT
---------------------------
Final Verdict: {'HIGH FRAUD RISK (ESCALATE TO SIU)' if pred == 1 else 'LOW RISK (APPROVE SETTLEMENT)'}
Estimated Fraud Risk: {fraud_pct:.2f}%
Genuine Confidence: {genuine_pct:.2f}%
Consensus Vote: {fraud_votes}/5 Models flagged Fraud

2. DRIVER & POLICY PROFILE
---------------------------
Age of Driver: {age_of_driver} yrs | Gender: {gender} | Marital: {marital_status}
Annual Income: ${annual_income:,.2f}
Completed Higher Education: {high_education}
Address Change (1-5 yrs): {address_change}
Policy Deductible: ${policy_deductible} | Annual Premium: ${annual_premium:,.2f}
Zip Code: {zip_code} | Property: {property_status}

3. VEHICLE & INCIDENT DETAILS
---------------------------
Vehicle: {age_of_vehicle} yrs old ({vehicle_category}, {vehicle_color}) | Price: ${vehicle_price:,.2f}
Claim Date: {claim_date} | Accident Site: {accident_site}
Witness Present: {witness_present} | Police Report Filed: {police_report}
Liability %: {liab_prct}% | Agent Channel: {channel}
Safety Rating: {safety_rating}/100 | Past Claims: {past_num_of_claims}
Days Open: {days_open} | Form Defects: {form_defects}

4. CLAIM FINANCIALS
---------------------------
Total Claim Amount: ${total_claim:,.2f}
Injury Claim Amount: ${injury_claim:,.2f}
Net Non-Injury Claim: ${(total_claim - injury_claim):,.2f}

=======================================================
CONFIDENTIAL • FOR INTERNAL AUDIT USE ONLY
=======================================================
"""
            st.download_button(
                label="📄 Download SIU Investigation Audit Dossier",
                data=dossier_text,
                file_name=f"SIU_Fraud_Audit_Case_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ═════════════════════════════════════════════════════════════════════════════
# 5. WHAT-IF SENSITIVITY SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "🔮 5. What-If Sensitivity Simulator":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #8A2387 0%, #E94057 50%, #F27121 100%);">
            <h1 class="hero-title">🔮 What-If Sensitivity & Risk Frontier Lab</h1>
            <p class="hero-subtitle">Dynamically perturb individual claim attributes to visualize how risk probabilities respond across decision boundaries.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_sim_cfg, col_sim_plot = st.columns([5, 7], gap="large")
    
    with col_sim_cfg:
        st.markdown("### 🎛️ Perturbation Controls")
        sim_model = st.selectbox("Select Model to Simulate:", list(pipelines.keys()), index=0)
        target_param = st.selectbox(
            "Feature to Perturb & Sweep:",
            [
                "Total Claim Amount ($)",
                "Age of Driver (years)",
                "Driver Safety Rating",
                "Form Defects Count",
                "Past Number of Claims",
                "Annual Income ($)"
            ]
        )
        
        st.markdown("---")
        st.markdown("##### Baseline Case Anchor Settings:")
        base_age = st.slider("Baseline Driver Age:", 18, 80, 35)
        base_income = st.slider("Baseline Annual Income ($):", 10000, 100000, 35000, step=5000)
        base_claim = st.slider("Baseline Total Claim ($):", 1000, 30000, 6000, step=1000)
        base_defects = st.slider("Baseline Form Defects:", 0, 15, 3)
        base_safety = st.slider("Baseline Safety Rating:", 0, 100, 70)
        base_past_claims = st.slider("Baseline Past Claims:", 0, 6, 1)

    with col_sim_plot:
        st.markdown("### 📈 Live Dynamic Risk Response Curve")
        
        sim_pipe = pipelines[sim_model]
        
        # Build baseline input template
        base_dict = {
            'age_of_driver': base_age, 'gender': 'Male', 'marital_status': 'Married',
            'annual_income': float(base_income), 'high_education': 'No', 'address_change': 'No',
            'policy_deductible': 500, 'annual_premium': 1000.0, 'zip_code': 50027,
            'property_status': 'Own', 'age_of_vehicle': 5, 'vehicle_category': 'Medium',
            'vehicle_price': 22000.0, 'vehicle_color': 'white', 'claim_date': date.today(),
            'accident_site': 'Local', 'witness_present': 'No', 'liab_prct': 50,
            'channel': 'Broker', 'police_report': 'No', 'safety_rating': base_safety,
            'past_num_of_claims': base_past_claims, 'days_open': 8.0, 'form_defects': base_defects,
            'injury_claim': 1000.0, 'total_claim': float(base_claim)
        }
        
        # Generate sweep range
        if target_param == "Total Claim Amount ($)":
            sweep_vals = np.linspace(1000, 35000, 40)
            x_label = "Total Claim Amount ($)"
        elif target_param == "Age of Driver (years)":
            sweep_vals = np.linspace(18, 80, 40)
            x_label = "Age of Driver"
        elif target_param == "Driver Safety Rating":
            sweep_vals = np.linspace(0, 100, 40)
            x_label = "Driver Safety Rating (0-100)"
        elif target_param == "Form Defects Count":
            sweep_vals = np.linspace(0, 20, 21)
            x_label = "Form Defects Count"
        elif target_param == "Past Number of Claims":
            sweep_vals = np.linspace(0, 8, 9)
            x_label = "Past Number of Claims"
        else:
            sweep_vals = np.linspace(5000, 120000, 40)
            x_label = "Annual Income ($)"
            
        prob_curve = []
        for val in sweep_vals:
            temp_dict = base_dict.copy()
            if target_param == "Total Claim Amount ($)":
                temp_dict['total_claim'] = val
                temp_dict['injury_claim'] = min(temp_dict['injury_claim'], val * 0.4)
            elif target_param == "Age of Driver (years)":
                temp_dict['age_of_driver'] = val
            elif target_param == "Driver Safety Rating":
                temp_dict['safety_rating'] = val
            elif target_param == "Form Defects Count":
                temp_dict['form_defects'] = val
            elif target_param == "Past Number of Claims":
                temp_dict['past_num_of_claims'] = int(val)
            else:
                temp_dict['annual_income'] = val
                
            inp_df = ml_engine.format_single_input(temp_dict)
            if hasattr(sim_pipe, "predict_proba"):
                p_val = sim_pipe.predict_proba(inp_df)[0][1] * 100
            else:
                p_val = 100.0 if sim_pipe.predict(inp_df)[0] == 1 else 0.0
            prob_curve.append(p_val)
            
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sweep_vals,
            y=prob_curve,
            mode='lines+markers',
            name='Fraud Probability %',
            line=dict(color='#ff5722', width=3.5),
            marker=dict(size=6, color='#ff5722')
        ))
        
        fig_sim.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% Classification Threshold")
        fig_sim.update_layout(
            title=f"Sensitivity Curve: {target_param} vs Fraud Risk %",
            xaxis_title=x_label,
            yaxis_title="Estimated Fraud Risk Probability (%)",
            yaxis_range=[0, 100],
            height=400
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
        st.markdown("""
            <div class="custom-card">
                <b>💡 Sensitivity Takeaway:</b>
                Notice how the probability curve non-linearly responds as individual parameters shift across tree splitting thresholds.
                This sensitivity curve enables underwriters to pinpoint exact break-even decision tipping points.
            </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 6. FINANCIAL IMPACT & ROI CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "💰 6. Financial Impact & ROI Calculator":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #134E5E 0%, #71B280 100%);">
            <h1 class="hero-title">💰 Financial Impact & SIU ROI Calculator</h1>
            <p class="hero-subtitle">Quantify direct cost savings, prevented fraudulent payouts, and return on investment (ROI) from machine learning deployment.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_roi_cfg, col_roi_res = st.columns([5, 7], gap="large")
    
    with col_roi_cfg:
        st.markdown("### 🏢 Portfolio Volume Assumptions")
        annual_claims = st.number_input("Annual Claims Volume Ingested:", min_value=1000, max_value=500000, value=25000, step=5000)
        avg_claim_cost = st.number_input("Average Payout per Claim ($):", min_value=1000.0, max_value=50000.0, value=7500.0, step=500.0)
        siu_audit_cost = st.number_input("SIU Audit Cost per Flagged Claim ($):", min_value=100.0, max_value=3000.0, value=450.0, step=50.0)
        
        st.markdown("---")
        st.markdown("### 🤖 Model Performance Selected:")
        roi_model = st.selectbox("Select Benchmark Model:", list(pipelines.keys()), index=0)
        
        m_row = comp_df[comp_df['Model'] == roi_model].iloc[0]
        recall_val = float(m_row['Recall'])
        precision_val = float(m_row['Precision'])
        fraud_prevalence = 0.2579  # 25.79% in dataset

    with col_roi_res:
        st.markdown("### 💵 Projected Business Return & Cost Savings")
        
        # Financial Mathematics
        total_fraud_claims = annual_claims * fraud_prevalence
        detected_fraud = total_fraud_claims * recall_val
        flagged_for_audit = (detected_fraud / precision_val) if precision_val > 0 else 0
        
        fraud_losses_prevented = detected_fraud * avg_claim_cost
        total_siu_investigation_costs = flagged_for_audit * siu_audit_cost
        net_annual_savings = fraud_losses_prevented - total_siu_investigation_costs
        roi_pct = (net_annual_savings / total_siu_investigation_costs * 100) if total_siu_investigation_costs > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Fraud Claims Intercepted", f"{int(detected_fraud):,} cases", f"{recall_val:.1%} Recall")
        with k2:
            st.metric("Gross Losses Prevented", f"${fraud_losses_prevented:,.0f}")
        with k3:
            st.metric("Net Annual Savings", f"${net_annual_savings:,.0f}", f"{roi_pct:.0f}% ROI")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Financial Comparison Waterfall / Bar
        fin_df = pd.DataFrame({
            'Category': ['Gross Fraud Prevented', 'SIU Investigation Overhead', 'Net Realized Savings'],
            'Amount ($)': [fraud_losses_prevented, total_siu_investigation_costs, net_annual_savings]
        })
        fig_fin = px.bar(
            fin_df,
            x='Category',
            y='Amount ($)',
            color='Category',
            text_auto='$,.0f',
            title=f"Annual Financial Impact Analysis ({roi_model})",
            color_discrete_map={
                'Gross Fraud Prevented': '#2ecc71',
                'SIU Investigation Overhead': '#e74c3c',
                'Net Realized Savings': '#00d2ff'
            }
        )
        st.plotly_chart(fig_fin, use_container_width=True)
        
        st.markdown(f"""
            <div class="custom-card">
                <b>📊 Executive Summary:</b><br>
                By deploying <b>{roi_model}</b> with a recall of <b>{recall_val:.1%}</b> across an annual portfolio of <b>{annual_claims:,} claims</b>, 
                your organization prevents an estimated <b>${fraud_losses_prevented:,.0f}</b> in fraudulent disbursements while spending <b>${total_siu_investigation_costs:,.0f}</b> in SIU investigations, 
                yielding a <b>Net Bottom-Line Profit of ${net_annual_savings:,.0f}</b> (ROI: <b>{roi_pct:,.1f}%</b>).
            </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 7. BATCH CLAIMS AUDIT (CSV PROCESSOR)
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "📂 7. Batch Claims Audit (CSV)":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);">
            <h1 class="hero-title">📂 Batch Claims Audit & CSV Processor</h1>
            <p class="hero-subtitle">Screen hundreds of claims in seconds with high-throughput multi-model batch inference.</p>
        </div>
    """, unsafe_allow_html=True)
    
    b_col1, b_col2 = st.columns([7, 5])
    with b_col1:
        batch_model_name = st.selectbox("Select Model for Batch Inference:", list(pipelines.keys()), index=0)
    with b_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample_batch = st.button("🧪 Load Sample Batch (20 Historical Claims)", use_container_width=True)
        
    uploaded_file = st.file_uploader("Upload Claims CSV File", type=["csv"])
    
    df_to_process = None
    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV with {df_to_process.shape[0]} rows and {df_to_process.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif load_sample_batch:
        df_to_process = clean_df.drop(columns=['fraud_reported'], errors='ignore').sample(20, random_state=42)
        st.info("Loaded 20 sample claims from dataset.")
        
    if df_to_process is not None:
        st.markdown("### 🔍 Processed Batch Results")
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
            
            total_b = len(batch_preds)
            fraud_b = int(sum(batch_preds))
            gen_b = int(total_b - fraud_b)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Batch Claims", f"{total_b:,}")
            k2.metric("Flagged High Risk", f"{fraud_b:,}", f"{(fraud_b/total_b)*100:.1f}%")
            k3.metric("Passed Genuine", f"{gen_b:,}", f"{(gen_b/total_b)*100:.1f}%")
            
            # Pie breakdown
            b_pie = px.pie(
                names=['Genuine Claims', 'Flagged Fraud'],
                values=[gen_b, fraud_b],
                color=['Genuine Claims', 'Flagged Fraud'],
                color_discrete_map={'Genuine Claims': '#2ecc71', 'Flagged Fraud': '#e74c3c'},
                title="Batch Classification Breakdown",
                hole=0.4
            )
            st.plotly_chart(b_pie, use_container_width=True)
            
            st.dataframe(results_batch, use_container_width=True)
            
            csv_data = results_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Annotated Predictions CSV",
                data=csv_data,
                file_name=f"fraud_predictions_{batch_model_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error executing batch inference: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# 8. PROJECT DOCUMENTATION & SOP PORTFOLIO
# ═════════════════════════════════════════════════════════════════════════════
elif app_mode == "📑 8. Project SOP & Documentation":
    st.markdown("""
        <div class="hero-banner" style="background: linear-gradient(135deg, #2C3E50 0%, #4CA1AF 100%);">
            <h1 class="hero-title">📑 Academic Documentation & SOP Portfolio</h1>
            <p class="hero-subtitle">Machine Learning Project — Complete 5-Week Syllabus Implementation & Theoretical Documentation.</p>
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
