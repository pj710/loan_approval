"""
Streamlit Dashboard for AI-Powered Mortgage Underwriting Assistant.

Author: Josiah Gordor
Date: February 2026

Features:
- Loan application input form
- Real-time predictions with confidence scores
- SHAP-based explanation panel
- Fairness metrics dashboard
- What-if analysis
- Batch upload processing

Run:
    streamlit run src/dashboard/app.py
"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
from pathlib import Path
import io
import yaml
from dotenv import load_dotenv
import os

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Load config.yaml
config_path = Path(__file__).parent.parent.parent / 'config.yaml'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {}

API_URL = os.getenv('API_URL')
if not API_URL and 'api' in config:
    API_URL = f"http://{config['api']['host']}:{config['api']['port']}"
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Page configuration
st.set_page_config(
    page_title="Loan Approval Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f77b4 0%, #2ecc71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e9ecef;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Status colors */
    .approved {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .denied {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .low-risk { color: #28a745; font-weight: 600; }
    .moderate-risk { color: #ffc107; font-weight: 600; }
    .high-risk { color: #dc3545; font-weight: 600; }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button improvements */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Input fields */
    .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 8px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Author footer */
    .author-footer {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        margin-top: 2rem;
    }
    .author-footer a {
        color: #ffd700;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health() -> dict:
    """Check if the API is available and models are loaded."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException:
        return {"status": "unavailable", "model_loaded": False}


def get_prediction(application: dict) -> dict:
    """Get prediction from API."""
    try:
        response = requests.post(f"{API_URL}/predict", json=application, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Prediction failed")}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_explanation(application: dict) -> dict:
    """Get SHAP explanation from API."""
    try:
        response = requests.post(f"{API_URL}/explain", json=application, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Explanation failed")}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def create_gauge_chart(probability: float, title: str = "Approval Probability") -> go.Figure:
    """Create a gauge chart for probability display."""
    color = "#28a745" if probability >= 0.5 else "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 50], 'color': '#ffffcc'},
                {'range': [50, 70], 'color': '#ccffcc'},
                {'range': [70, 100], 'color': '#99ff99'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_feature_importance_chart(contributions: list) -> go.Figure:
    """Create horizontal bar chart for feature importance."""
    if not contributions:
        return None
    
    # Helper to safely convert shap_value to float
    def get_shap_float(x):
        val = x.get('shap_value', 0)
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0
    
    # Sort by absolute SHAP value
    sorted_contrib = sorted(contributions, key=lambda x: abs(get_shap_float(x)), reverse=True)[:10]
    
    features = [c.get('feature', 'Unknown') for c in sorted_contrib]
    values = [get_shap_float(c) for c in sorted_contrib]
    colors = ['#28a745' if v >= 0 else '#dc3545' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Contributions (SHAP Values)",
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=150, r=50, t=50, b=50),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_risk_meter(risk_level: str) -> str:
    """Create HTML risk meter."""
    risk_colors = {
        "Low Risk": ("#28a745", "🟢"),
        "Moderate Risk": ("#ffc107", "🟡"),
        "High Risk": ("#dc3545", "🔴"),
        "Unknown": ("#6c757d", "⚪")
    }
    color, emoji = risk_colors.get(risk_level, ("#6c757d", "⚪"))
    return f'<span style="color: {color}; font-size: 1.5rem;">{emoji} {risk_level}</span>'


def load_fairness_metrics() -> pd.DataFrame:
    """Load latest fairness metrics from results."""
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return pd.DataFrame()
    
    fairness_files = sorted(results_dir.glob("fairness_metrics_*.csv"), reverse=True)
    if fairness_files:
        return pd.read_csv(fairness_files[0])
    return pd.DataFrame()


def load_fair_model_performance() -> pd.DataFrame:
    """Load fair model performance comparison data."""
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return pd.DataFrame()
    
    perf_files = sorted(results_dir.glob("fair_model_performance_*.csv"), reverse=True)
    if perf_files:
        return pd.read_csv(perf_files[0])
    return pd.DataFrame()


def load_fair_model_fairness() -> pd.DataFrame:
    """Load fair model fairness comparison data."""
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return pd.DataFrame()
    
    fair_files = sorted(results_dir.glob("fair_model_fairness_*.csv"), reverse=True)
    if fair_files:
        return pd.read_csv(fair_files[0])
    return pd.DataFrame()


def process_batch_applications(df: pd.DataFrame) -> pd.DataFrame:
    """Process batch applications through API."""
    results = []
    
    required_cols = ['loan_amount', 'property_value', 'income', 'interest_rate', 'loan_term']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return pd.DataFrame()
    
    progress_bar = st.progress(0)
    for idx, row in df.iterrows():
        application = {
            "loan_amount": float(row['loan_amount']),
            "property_value": float(row['property_value']),
            "income": float(row['income']),
            "interest_rate": float(row['interest_rate']),
            "loan_term": float(row.get('loan_term', 360))
        }
        
        # Add optional fields
        if 'is_fha_loan' in row:
            application['is_fha_loan'] = bool(row['is_fha_loan'])
        if 'is_va_loan' in row:
            application['is_va_loan'] = bool(row['is_va_loan'])
        if 'has_coborrower' in row:
            application['has_coborrower'] = bool(row['has_coborrower'])
        
        # Add selected model
        application['model_name'] = st.session_state.get('selected_model', 'xgb_fair')
        
        pred = get_prediction(application)
        
        results.append({
            'application_id': idx + 1,
            **application,
            'prediction': pred.get('prediction', 'Error'),
            'probability': pred.get('probability', 0),
            'risk_level': pred.get('risk_level', 'Unknown'),
            'confidence': pred.get('confidence', 0)
        })
        
        progress_bar.progress((idx + 1) / len(df))
    
    return pd.DataFrame(results)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/home-mortgage.png", width=80)
    st.markdown("### 🏦 Loan Assistant")
    st.caption("AI-Powered Mortgage Underwriting")
    
    st.divider()
    
    page = st.radio(
        "📍 Navigation",
        ["🏠 Single Application", "📊 What-If Analysis", "📁 Batch Upload", "⚖️ Fairness Dashboard"],
        index=0
    )
    
    st.divider()
    
    # API Health Status
    st.markdown("**🔌 System Status**")
    health = check_api_health()
    
    if health.get("status") == "healthy":
        st.success("✅ API Online")
        st.caption(health.get('model_name', 'Unknown'))
    elif health.get("status") == "unhealthy":
        st.warning("⚠️ Models Loading...")
        st.caption("Some features may be limited")
    else:
        st.error("❌ API Offline")
        st.caption("Start the API server:")
        st.code("uvicorn src.api.main:app --port 8000")
    
    st.divider()
    
    # Model Selection
    st.subheader("🤖 Model Selection")
    FAIR_MODELS = {
        "XGBoost Fair": "xgb_fair",
        "Random Forest Fair": "rf_fair",
        "Logistic Regression Fair": "lr_fair",
        "Neural Network Fair": "nn_fair",
        "GLM Fair": "glm_fair",
        "FasterRisk Fair": "fasterrisk_fair",
        "GOSDT Fair": "gosdt_fair"
    }
    selected_model_display = st.selectbox(
        "Select Fair Model",
        options=list(FAIR_MODELS.keys()),
        index=0,
        help="Choose which fair ML model to use for predictions"
    )
    selected_model_key = FAIR_MODELS[selected_model_display]
    st.session_state['selected_model'] = selected_model_key
    
    st.divider()
    
    # Author info in sidebar
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 8px;'>
        <p style='margin: 0; color: #888; font-size: 11px;'>Developed by</p>
        <p style='margin: 5px 0 0 0; color: #4ecdc4; font-weight: 600;'>Josiah Gordor</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"v1.0.0 | {datetime.now().strftime('%Y-%m-%d')}")


# ============================================================================
# SINGLE APPLICATION PAGE
# ============================================================================

if page == "🏠 Single Application":
    st.markdown('<h1 class="main-header">🏠 AI Mortgage Underwriting Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Submit a loan application for instant AI-powered approval recommendation.", unsafe_allow_html=True)
    
    # Input Form
    st.subheader("📝 Loan Application Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Loan Information**")
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=10000,
            max_value=10000000,
            value=300000,
            step=10000,
            help="Total loan amount requested"
        )
        
        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.1,
            max_value=20.0,
            value=6.5,
            step=0.125,
            help="Annual interest rate"
        )
        
        loan_term = st.selectbox(
            "Loan Term",
            options=[180, 240, 360],
            index=2,
            format_func=lambda x: f"{x//12} years ({x} months)"
        )
    
    with col2:
        st.markdown("**Property & Income**")
        property_value = st.number_input(
            "Property Value ($)",
            min_value=10000,
            max_value=20000000,
            value=400000,
            step=10000,
            help="Appraised property value"
        )
        
        income = st.number_input(
            "Annual Income ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=5000,
            help="Gross annual income"
        )
        
        state_code = st.selectbox(
            "State",
            options=[
                "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
            ],
            index=4,  # Default to CA
            help="Select applicant's state"
        )
    
    with col3:
        st.markdown("**Loan Type & Options**")
        is_fha_loan = st.checkbox("FHA Insured Loan", value=False)
        is_va_loan = st.checkbox("VA Guaranteed Loan", value=False)
        has_coborrower = st.checkbox("Has Co-Borrower", value=False)
        
        applicant_age = st.number_input(
            "Applicant Age",
            min_value=18,
            max_value=100,
            value=35
        )
    
    # Computed metrics display
    ltv = loan_amount / property_value * 100 if property_value > 0 else 0
    lti = loan_amount / income if income > 0 else 0
    
    st.markdown("---")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Loan-to-Value (LTV)", f"{ltv:.1f}%", 
                  delta="Good" if ltv <= 80 else ("Fair" if ltv <= 95 else "High"),
                  delta_color="normal" if ltv <= 80 else ("off" if ltv <= 95 else "inverse"))
    
    with metric_col2:
        st.metric("Loan-to-Income (LTI)", f"{lti:.2f}x",
                  delta="Good" if lti <= 4 else ("Fair" if lti <= 6 else "High"),
                  delta_color="normal" if lti <= 4 else ("off" if lti <= 6 else "inverse"))
    
    with metric_col3:
        est_monthly = (loan_amount * (interest_rate/100/12) * (1 + interest_rate/100/12)**loan_term) / \
                      ((1 + interest_rate/100/12)**loan_term - 1) if interest_rate > 0 else loan_amount / loan_term
        st.metric("Est. Monthly Payment", f"${est_monthly:,.0f}")
    
    with metric_col4:
        dti_approx = (est_monthly * 12 / income * 100) if income > 0 else 0
        st.metric("Est. DTI Ratio", f"{dti_approx:.1f}%",
                  delta="Good" if dti_approx <= 36 else ("Fair" if dti_approx <= 43 else "High"),
                  delta_color="normal" if dti_approx <= 36 else ("off" if dti_approx <= 43 else "inverse"))
    
    st.markdown("---")
    
    # Submit button
    if st.button("🚀 Get AI Recommendation", type="primary", use_container_width=True):
        
        application = {
            "loan_amount": loan_amount,
            "property_value": property_value,
            "income": income,
            "interest_rate": interest_rate,
            "loan_term": loan_term,
            "state_code": state_code,
            "is_fha_loan": is_fha_loan,
            "is_va_loan": is_va_loan,
            "has_coborrower": has_coborrower,
            "applicant_age": applicant_age,
            "model_name": st.session_state.get('selected_model', 'xgb_fair')
        }
        
        with st.spinner("Analyzing application..."):
            prediction = get_prediction(application)
            explanation = get_explanation(application)
        
        if "error" in prediction:
            st.error(f"Prediction failed: {prediction['error']}")
            st.info("Make sure the API server is running: `uvicorn src.api.main:app --port 8000`")
        else:
            # Results section
            st.subheader("📋 AI Recommendation")
            
            result_col1, result_col2 = st.columns([1, 2])
            
            with result_col1:
                # Gauge chart
                fig = create_gauge_chart(prediction.get('probability', 0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Decision
                decision = prediction.get('prediction', 'Unknown')
                if decision == "Approved":
                    st.success(f"### ✅ {decision}")
                else:
                    st.error(f"### ❌ {decision}")
                
                # Risk level
                risk_level = prediction.get('risk_level', 'Unknown')
                st.markdown(create_risk_meter(risk_level), unsafe_allow_html=True)
                
                # Confidence
                confidence = prediction.get('confidence', 0)
                st.metric("Confidence Level", f"{confidence:.1f}%")
                
                # Processing time
                proc_time = prediction.get('processing_time_ms', 0)
                st.caption(f"Processed in {proc_time:.1f}ms")
            
            with result_col2:
                # Explanation panel
                if "error" not in explanation:
                    st.markdown("### 📊 Key Factors")
                    
                    # Feature importance chart
                    contributions = explanation.get('feature_contributions', [])
                    if contributions:
                        fig = create_feature_importance_chart(contributions)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Text explanation
                    explanation_text = explanation.get('explanation_text', '')
                    if explanation_text:
                        st.info(f"**Analysis:** {explanation_text}")
                    
                    # Top factors
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.markdown("**Positive Factors** 👍")
                        for factor in explanation.get('top_positive_factors', [])[:3]:
                            shap_val = float(factor.get('shap_value', 0)) if factor.get('shap_value') else 0.0
                            st.write(f"• {factor.get('feature')}: +{shap_val:.3f}")
                    
                    with col_neg:
                        st.markdown("**Negative Factors** 👎")
                        for factor in explanation.get('top_negative_factors', [])[:3]:
                            shap_val = float(factor.get('shap_value', 0)) if factor.get('shap_value') else 0.0
                            st.write(f"• {factor.get('feature')}: {shap_val:.3f}")


# ============================================================================
# WHAT-IF ANALYSIS PAGE
# ============================================================================

elif page == "📊 What-If Analysis":
    st.markdown('<h1 class="main-header">📊 What-If Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Explore how changing loan parameters affects approval probability.")
    
    st.subheader("Adjust Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_loan = st.number_input("Base Loan Amount ($)", value=300000, step=10000)
        base_property = st.number_input("Property Value ($)", value=400000, step=10000)
        base_income = st.number_input("Annual Income ($)", value=100000, step=5000)
    
    with col2:
        base_rate = st.number_input("Interest Rate (%)", value=6.5, step=0.25)
        base_term = st.selectbox("Loan Term", [180, 240, 360], index=2)
        has_coborrower = st.checkbox("Has Co-Borrower", value=False)
    
    st.markdown("---")
    
    st.subheader("Parameter Sensitivity Analysis")
    
    analysis_param = st.selectbox(
        "Select parameter to analyze",
        ["Loan-to-Value Ratio", "Loan-to-Income Ratio", "Interest Rate", "Loan Amount"]
    )
    
    if st.button("🔬 Run Analysis", type="primary"):
        
        results = []
        
        with st.spinner("Running sensitivity analysis..."):
            
            if analysis_param == "Loan-to-Value Ratio":
                # Vary LTV from 60% to 100%
                ltv_values = np.arange(0.60, 1.01, 0.05)
                for ltv in ltv_values:
                    loan_amt = ltv * base_property
                    app = {
                        "loan_amount": loan_amt,
                        "property_value": base_property,
                        "income": base_income,
                        "interest_rate": base_rate,
                        "loan_term": base_term
                    }
                    pred = get_prediction(app)
                    results.append({
                        "parameter": f"{ltv*100:.0f}%",
                        "value": ltv * 100,
                        "probability": pred.get("probability", 0) * 100
                    })
            
            elif analysis_param == "Loan-to-Income Ratio":
                # Vary LTI from 2x to 8x
                lti_values = np.arange(2, 8.5, 0.5)
                for lti in lti_values:
                    loan_amt = lti * base_income
                    app = {
                        "loan_amount": loan_amt,
                        "property_value": base_property,
                        "income": base_income,
                        "interest_rate": base_rate,
                        "loan_term": base_term
                    }
                    pred = get_prediction(app)
                    results.append({
                        "parameter": f"{lti:.1f}x",
                        "value": lti,
                        "probability": pred.get("probability", 0) * 100
                    })
            
            elif analysis_param == "Interest Rate":
                # Vary rate from 4% to 10%
                rate_values = np.arange(4, 10.5, 0.5)
                for rate in rate_values:
                    app = {
                        "loan_amount": base_loan,
                        "property_value": base_property,
                        "income": base_income,
                        "interest_rate": rate,
                        "loan_term": base_term
                    }
                    pred = get_prediction(app)
                    results.append({
                        "parameter": f"{rate:.1f}%",
                        "value": rate,
                        "probability": pred.get("probability", 0) * 100
                    })
            
            else:  # Loan Amount
                # Vary loan amount
                loan_values = np.arange(100000, 800000, 50000)
                for loan in loan_values:
                    app = {
                        "loan_amount": loan,
                        "property_value": base_property,
                        "income": base_income,
                        "interest_rate": base_rate,
                        "loan_term": base_term
                    }
                    pred = get_prediction(app)
                    results.append({
                        "parameter": f"${loan/1000:.0f}K",
                        "value": loan,
                        "probability": pred.get("probability", 0) * 100
                    })
        
        if results:
            df = pd.DataFrame(results)
            
            # Create chart
            fig = px.line(
                df,
                x="value",
                y="probability",
                markers=True,
                title=f"Approval Probability vs {analysis_param}"
            )
            
            fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% Threshold")
            
            fig.update_layout(
                xaxis_title=analysis_param,
                yaxis_title="Approval Probability (%)",
                yaxis_range=[0, 100],
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("Detailed Results")
            st.dataframe(df)


# ============================================================================
# BATCH UPLOAD PAGE
# ============================================================================

elif page == "📁 Batch Upload":
    st.markdown('<h1 class="main-header">📁 Batch Application Processing</h1>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file with multiple loan applications for batch processing.")
    
    # Template download
    st.subheader("📥 Download Template")
    
    template_df = pd.DataFrame({
        'loan_amount': [300000, 250000, 450000],
        'property_value': [400000, 300000, 600000],
        'income': [100000, 85000, 150000],
        'interest_rate': [6.5, 7.0, 6.25],
        'loan_term': [360, 360, 240],
        'is_fha_loan': [False, True, False],
        'is_va_loan': [False, False, False],
        'has_coborrower': [True, False, True]
    })
    
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="📄 Download CSV Template",
        data=csv_template,
        file_name="loan_application_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # File upload
    st.subheader("📤 Upload Applications")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: loan_amount, property_value, income, interest_rate, loan_term"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} applications")
            
            # Preview
            st.subheader("Preview")
            st.dataframe(df.head(10))
            
            # Process button
            if st.button("🚀 Process All Applications", type="primary"):
                
                results_df = process_batch_applications(df)
                
                if not results_df.empty:
                    st.success(f"✅ Processed {len(results_df)} applications")
                    
                    # Summary metrics
                    st.subheader("📊 Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    approved = (results_df['prediction'] == 'Approved').sum()
                    denied = (results_df['prediction'] == 'Denied').sum()
                    
                    with col1:
                        st.metric("Total Applications", len(results_df))
                    with col2:
                        st.metric("Approved", approved, delta=f"{approved/len(results_df)*100:.1f}%")
                    with col3:
                        st.metric("Denied", denied, delta=f"{denied/len(results_df)*100:.1f}%", delta_color="inverse")
                    with col4:
                        high_risk = (results_df['risk_level'] == 'High Risk').sum()
                        st.metric("High Risk", high_risk, delta_color="inverse")
                    
                    # Risk distribution chart
                    fig = px.pie(
                        results_df,
                        names='risk_level',
                        title='Risk Level Distribution',
                        color='risk_level',
                        color_discrete_map={
                            'Low Risk': '#28a745',
                            'Moderate Risk': '#ffc107',
                            'High Risk': '#dc3545'
                        }
                    )
                    st.plotly_chart(fig)
                    
                    # Detailed results
                    st.subheader("📋 Detailed Results")
                    
                    # Color-code results
                    def highlight_prediction(val):
                        if val == 'Approved':
                            return 'background-color: #d4edda'
                        elif val == 'Denied':
                            return 'background-color: #f8d7da'
                        return ''
                    
                    styled_df = results_df.style.applymap(
                        highlight_prediction,
                        subset=['prediction']
                    )
                    st.dataframe(styled_df)
                    
                    # Download results
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_results,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")


# ============================================================================
# FAIRNESS DASHBOARD PAGE
# ============================================================================

elif page == "⚖️ Fairness Dashboard":
    st.markdown('<h1 class="main-header">⚖️ Fair Model Comparison Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Compare performance and fairness metrics across all fair ML models.")
    
    # Load fair model comparison data
    perf_df = load_fair_model_performance()
    fair_df = load_fair_model_fairness()
    
    if perf_df.empty and fair_df.empty:
        st.warning("No fair model comparison data found. Run the model training notebooks to generate metrics.")
        st.info("Expected files: `results/fair_model_performance_*.csv` and `results/fair_model_fairness_*.csv`")
    else:
        # Performance Comparison Section
        if not perf_df.empty:
            st.subheader("📊 Model Performance Comparison")
            st.markdown("Compare accuracy, AUC, precision, recall, and F1 scores across all fair models.")
            
            # Performance metrics bar chart
            perf_metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']
            available_metrics = [m for m in perf_metrics if m in perf_df.columns]
            
            if available_metrics and 'Model' in perf_df.columns:
                # Filter to only include fair models (exclude Ensemble if desired)
                fair_model_names = ['LR_Fair', 'XGB_Fair', 'RF_Fair', 'NN_Fair', 'GLM_Fair', 'FasterRisk_Fair', 'GOSDT_Fair']
                perf_filtered = perf_df[perf_df['Model'].isin(fair_model_names)].copy()
                
                # Melt for grouped bar chart
                perf_melted = perf_filtered.melt(
                    id_vars=['Model'], 
                    value_vars=available_metrics,
                    var_name='Metric', 
                    value_name='Score'
                )
                
                fig_perf = px.bar(
                    perf_melted,
                    x='Model',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    title='Performance Metrics by Fair Model',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_perf.update_layout(
                    height=450,
                    xaxis_title="Model",
                    yaxis_title="Score",
                    yaxis_range=[0, 1.05],
                    legend_title="Metric"
                )
                fig_perf.add_hline(y=0.95, line_dash="dash", line_color="green", 
                                   annotation_text="95% Target", annotation_position="top right")
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Best model highlights
                st.markdown("**🏆 Best Models by Metric:**")
                best_cols = st.columns(len(available_metrics))
                for i, metric in enumerate(available_metrics):
                    best_idx = perf_filtered[metric].idxmax()
                    best_model = perf_filtered.loc[best_idx, 'Model']
                    best_score = perf_filtered.loc[best_idx, metric]
                    with best_cols[i]:
                        st.metric(metric, best_model.replace('_Fair', ''), f"{best_score:.3f}")
        
        st.markdown("---")
        
        # Fairness Comparison Section
        if not fair_df.empty:
            st.subheader("⚖️ Fairness Metrics Comparison")
            st.markdown("Compare fairness across protected attributes for all fair models.")
            
            # Group fairness by model and average across attributes
            if 'Model' in fair_df.columns:
                fair_model_names = ['LR_Fair', 'XGB_Fair', 'RF_Fair', 'NN_Fair', 'GLM_Fair', 'FasterRisk_Fair', 'GOSDT_Fair']
                fair_filtered = fair_df[fair_df['Model'].isin(fair_model_names)].copy()
                
                # Aggregate fairness metrics by model
                fairness_agg = fair_filtered.groupby('Model').agg({
                    'DPD': 'mean',
                    'EOD': 'mean',
                    'Disparate_Impact': 'mean'
                }).reset_index()
                
                # Convert DPD/EOD to pass rate (1 - disparity for better visualization)
                fairness_agg['Demographic_Parity'] = 1 - fairness_agg['DPD']
                fairness_agg['Equalized_Odds'] = 1 - fairness_agg['EOD'].fillna(0)
                
                # Fairness metrics bar chart
                fairness_melted = fairness_agg.melt(
                    id_vars=['Model'],
                    value_vars=['Demographic_Parity', 'Equalized_Odds', 'Disparate_Impact'],
                    var_name='Metric',
                    value_name='Score'
                )
                
                fig_fair = px.bar(
                    fairness_melted,
                    x='Model',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    title='Fairness Metrics by Fair Model (Higher is Better)',
                    color_discrete_sequence=['#2ecc71', '#3498db', '#9b59b6']
                )
                fig_fair.update_layout(
                    height=450,
                    xaxis_title="Model",
                    yaxis_title="Score",
                    yaxis_range=[0, 1.1],
                    legend_title="Metric"
                )
                fig_fair.add_hline(y=0.8, line_dash="dash", line_color="orange", 
                                   annotation_text="80% Threshold", annotation_position="top right")
                fig_fair.add_hline(y=1.0, line_dash="dot", line_color="green", 
                                   annotation_text="Perfect Fairness")
                st.plotly_chart(fig_fair, use_container_width=True)
                
                # Pass rates by protected attribute
                st.subheader("📋 Fairness Pass Rates by Protected Attribute")
                
                if 'Protected_Attribute' in fair_filtered.columns:
                    attr_col = 'Protected_Attribute'
                else:
                    attr_col = fair_filtered.columns[1] if len(fair_filtered.columns) > 1 else None
                
                if attr_col:
                    # Count passes by model
                    pass_cols = [c for c in fair_filtered.columns if 'Pass' in c]
                    if pass_cols:
                        pass_summary = fair_filtered.groupby('Model')[pass_cols].sum().reset_index()
                        
                        pass_melted = pass_summary.melt(
                            id_vars=['Model'],
                            value_vars=pass_cols,
                            var_name='Criterion',
                            value_name='Pass Count'
                        )
                        
                        fig_pass = px.bar(
                            pass_melted,
                            x='Model',
                            y='Pass Count',
                            color='Criterion',
                            barmode='stack',
                            title='Fairness Criteria Pass Count by Model',
                            color_discrete_sequence=['#27ae60', '#2980b9', '#8e44ad']
                        )
                        fig_pass.update_layout(height=400)
                        st.plotly_chart(fig_pass, use_container_width=True)
        
        st.markdown("---")
        
        # Fairness thresholds explanation
        st.subheader("ℹ️ Fairness Criteria")
        
        st.markdown("""
        | Metric | Threshold | Description |
        |--------|-----------|-------------|
        | **Demographic Parity** | ≥ 0.80 | Approval rates should be similar across groups |
        | **Equalized Odds** | ≥ 0.80 | True positive and false positive rates should be similar |
        | **Disparate Impact** | 0.80 - 1.25 | Ratio of approval rates between groups |
        
        **Regulatory Context**: Fair lending regulations (ECOA, Fair Housing Act) require that lending decisions 
        do not discriminate based on protected characteristics including race, ethnicity, sex, or age.
        """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div class="author-footer">
        <strong>AI-Powered Mortgage Underwriting Assistant</strong><br>
        Built with Streamlit & FastAPI | Fair ML Models<br>
        <span style="font-size: 0.9rem;">👨‍💻 Developed by <strong>Josiah Gordor</strong></span><br>
        <span style="font-size: 0.75rem;">Data: HMDA 2024 (All US States) | © 2026</span>
    </div>
    """,
    unsafe_allow_html=True
)
