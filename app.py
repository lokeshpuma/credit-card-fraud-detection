import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit Card Fraud Guard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN STYLING INJECTION (DARK GLOW THEME) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0B0E14;
        color: #F3F4F6;
    }
    
    /* Header Gradient */
    .glowing-header {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.05em;
    }
    
    .glowing-subtitle {
        color: #9CA3AF;
        text-align: center;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Glassmorphic cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Status Glow Cards */
    .status-safe {
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 20px;
        color: #34D399;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
        animation: pulse-green 2s infinite alternate;
    }
    
    .status-fraud {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        padding: 20px;
        color: #F87171;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
        animation: pulse-red 1.5s infinite alternate;
    }

    @keyframes pulse-green {
        0% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.1); }
        100% { box-shadow: 0 0 25px rgba(16, 185, 129, 0.3); }
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.15); }
        100% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.4); }
    }
    
    /* Interactive Metric styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Button custom hover effects */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        border: none;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    }

    div.stButton > button:first-child:active {
        transform: translateY(1px);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL WORKFLOW ---
@st.cache_resource
def load_prediction_model():
    try:
        with open("credit_card_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading prediction model: {e}")
        return None

model = load_prediction_model()

# --- PRESET TRANSACTIONS ---
PRESETS = {
    "Safe Transaction (Sample #1)": {
        'Time': 0, 'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155, 
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698, 'V9': 0.363787, 
        'V10': 0.090794, 'V11': -0.551599, 'V12': -0.617801, 'V13': -0.99139, 'V14': -0.311169, 
        'V15': 1.468177, 'V16': -0.470401, 'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 
        'V20': 0.251412, 'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928, 
        'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053, 'Amount': 149.62
    },
    "Safe Transaction (Sample #2)": {
        'Time': 0, 'V1': 1.191857, 'V2': 0.266151, 'V3': 0.16648, 'V4': 0.448154, 
        'V5': 0.060018, 'V6': -0.082361, 'V7': -0.078803, 'V8': 0.085102, 'V9': -0.255425, 
        'V10': -0.166974, 'V11': 1.612727, 'V12': 1.065235, 'V13': 0.489095, 'V14': -0.143772, 
        'V15': 0.635558, 'V16': 0.463917, 'V17': -0.114805, 'V18': -0.183361, 'V19': -0.145783, 
        'V20': -0.069083, 'V21': -0.225775, 'V22': -0.638672, 'V23': 0.101288, 'V24': -0.339846, 
        'V25': 0.16717, 'V26': 0.125895, 'V27': -0.008983, 'V28': 0.014724, 'Amount': 2.69
    },
    "Fraudulent Transaction (Sample #1)": {
        'Time': 406, 'V1': -2.312227, 'V2': 1.951992, 'V3': -1.609851, 'V4': 3.997906, 
        'V5': -0.522188, 'V6': -1.426545, 'V7': -2.537387, 'V8': 1.391657, 'V9': -2.770089, 
        'V10': -2.772272, 'V11': 3.202033, 'V12': -2.899907, 'V13': -0.595222, 'V14': -4.289254, 
        'V15': 0.389724, 'V16': -1.140747, 'V17': -2.830056, 'V18': -0.016822, 'V19': 0.416956, 
        'V20': 0.126911, 'V21': 0.517232, 'V22': -0.035049, 'V23': -0.465211, 'V24': 0.320198, 
        'V25': 0.044519, 'V26': 0.17784, 'V27': 0.261145, 'V28': -0.143276, 'Amount': 0.0
    },
    "Fraudulent Transaction (Sample #2)": {
        'Time': 472, 'V1': -3.043541, 'V2': -3.157307, 'V3': 1.088463, 'V4': 2.288644, 
        'V5': 1.359805, 'V6': -1.064823, 'V7': 0.325574, 'V8': -0.067794, 'V9': -0.270953, 
        'V10': -0.838587, 'V11': -0.414575, 'V12': -0.503141, 'V13': 0.676502, 'V14': -1.692029, 
        'V15': 2.000635, 'V16': 0.66678, 'V17': 0.599717, 'V18': 1.725321, 'V19': 0.283345, 
        'V20': 2.102339, 'V21': 0.661696, 'V22': 0.435477, 'V23': 1.375966, 'V24': -0.293803, 
        'V25': 0.279798, 'V26': -0.145362, 'V27': -0.252773, 'V28': 0.035764, 'Amount': 529.0
    }
}

# --- HEADER SECTION ---
st.markdown("<h1 class='glowing-header'>💳 Credit Card Fraud Guard</h1>", unsafe_allow_html=True)
st.markdown("<p class='glowing-subtitle'>An advanced Machine Learning solution for real-time transaction risk scoring and anomaly detection.</p>", unsafe_allow_html=True)

# Initialize Session State values if not present
if "manual_features" not in st.session_state:
    st.session_state["manual_features"] = PRESETS["Safe Transaction (Sample #1)"].copy()

# --- MAIN NAVIGATION TABS ---
tab_single, tab_bulk = st.tabs(["🔍 Single-Transaction Scanner", "📁 Bulk CSV Processor"])

# ==============================================================================
# TAB 1: SINGLE TRANSACTION SCANNER
# ==============================================================================
with tab_single:
    st.markdown("""
    <div class='glass-card'>
        <h3>⚡ Live Scanner Sandbox</h3>
        <p style='color: #9CA3AF;'>Perform high-fidelity security analysis on specific credit card transactions. 
        You can fine-tune individual PCA components manually or quickly pre-load standard templates using the selector below.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Preset Selector Box
    col_preset, col_empty = st.columns([2, 2])
    with col_preset:
        selected_preset = st.selectbox(
            "📋 Quick-Load Verified Transaction Presets",
            options=["-- Manual Custom Entry --"] + list(PRESETS.keys()),
            index=0
        )
        
        if selected_preset != "-- Manual Custom Entry --":
            # Update state with preset
            st.session_state["manual_features"] = PRESETS[selected_preset].copy()
            st.toast(f"Loaded: {selected_preset}", icon="⚡")
            
    # Form Setup
    with st.form("manual_scanner_form"):
        st.markdown("#### 🔹 Basic Transaction Metadata")
        col_time, col_amount = st.columns(2)
        
        with col_time:
            time_val = st.number_input(
                "Transaction Time Offset (seconds from epoch start)",
                min_value=0,
                value=int(st.session_state["manual_features"]['Time']),
                help="Time elapsed in seconds since the first recorded transaction."
            )
            
        with col_amount:
            amount_val = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                value=float(st.session_state["manual_features"]['Amount']),
                format="%.2f",
                help="Total purchase amount of the transaction."
            )
            
        st.markdown("---")
        st.markdown("#### 🧬 PCA Transformed Dimensionality Components (V1 - V28)")
        
        # Grid of V1-V28 (4 components per row)
        pca_cols = st.columns(4)
        pca_inputs = {}
        
        for idx in range(1, 29):
            col_target = pca_cols[(idx - 1) % 4]
            var_name = f"V{idx}"
            with col_target:
                pca_inputs[var_name] = st.number_input(
                    f"Component {var_name}",
                    value=float(st.session_state["manual_features"][var_name]),
                    format="%.6f"
                )
                
        # Form Submit
        submit_btn = st.form_submit_button("🔍 Run Fraud Risk Scan")
        
        if submit_btn:
            if model is None:
                st.error("Model could not be retrieved. Please check system logs.")
            else:
                # Construct feature vector matching order of feature_names_in_
                feature_names = [
                    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
                ]
                
                inputs = {'Time': time_val, 'Amount': amount_val}
                inputs.update(pca_inputs)
                
                ordered_vector = [inputs[feat] for feat in feature_names]
                input_df = pd.DataFrame([ordered_vector], columns=feature_names)
                
                with st.spinner("Analyzing PCA pattern matrix & evaluating anomaly risk..."):
                    time.sleep(0.6) # Short delay for realistic AI scanner interaction feel
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    
                risk_pct = probabilities[1] * 100
                safe_pct = probabilities[0] * 100
                
                st.markdown("### 📊 Live Security Assessment Report")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class='status-fraud'>
                        <h2 style='color:#F87171; margin-top:0;'>🚨 CRITICAL WARNING: HIGH FRAUDULENT RISK</h2>
                        <p style='color:#FFFFFF; font-size:1.1rem;'>This transaction displays characteristic patterns corresponding strongly to documented fraudulent activity.</p>
                        <hr style='border-color: rgba(239, 68, 68, 0.3);'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <span class='metric-label'>FRAUD SUSPICION PROBABILITY</span>
                                <div class='metric-value' style='color:#EF4444;'>{risk_pct:.2f}%</div>
                            </div>
                            <div>
                                <span class='metric-label'>STATUS ASSESSMENT</span>
                                <div class='metric-value' style='color:#EF4444;'>SHIELD BLOCKED</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='status-safe'>
                        <h2 style='color:#34D399; margin-top:0;'>🟢 TRANSACTION SIGNATURE VERIFIED (SAFE)</h2>
                        <p style='color:#FFFFFF; font-size:1.1rem;'>This transaction aligns perfectly with typical user behavior parameters. Anomaly index is within standard nominal parameters.</p>
                        <hr style='border-color: rgba(16, 185, 129, 0.3);'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <span class='metric-label'>SAFE MATCH CONFIDENCE</span>
                                <div class='metric-value' style='color:#10B981;'>{safe_pct:.2f}%</div>
                            </div>
                            <div>
                                <span class='metric-label'>STATUS ASSESSMENT</span>
                                <div class='metric-value' style='color:#10B981;'>APPROVED</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: BULK CSV PROCESSOR
# ==============================================================================
with tab_bulk:
    st.markdown("""
    <div class='glass-card'>
        <h3>📁 High-Throughput Batch Scanner</h3>
        <p style='color: #9CA3AF;'>Upload a CSV batch of credit card transactions (containing column headers corresponding to Time, Amount, V1–V28) to run rapid parallel predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download templates helper
    col_dl, col_up = st.columns([1, 2])
    
    with col_dl:
        st.markdown("##### 📥 Fetch Sample Schema")
        st.write("Ensure your CSV file matches the 30-feature schema required by the model. Click below to download a test schema file ready for processing.")
        
        # Build template df
        sample_df = pd.DataFrame([PRESETS["Safe Transaction (Sample #1)"], PRESETS["Fraudulent Transaction (Sample #1)"]])
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Sample CSV Template",
            data=sample_csv,
            file_name="creditcard_sample_upload.csv",
            mime="text/csv"
        )
        
    with col_up:
        st.markdown("##### 📤 Upload Batch File")
        uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                
                # Check column headers match
                required_cols = [
                    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
                ]
                
                missing_cols = [c for c in required_cols if c not in data.columns]
                
                if missing_cols:
                    st.error(f"Failed schema check. The uploaded CSV is missing the following required features: {missing_cols}")
                else:
                    st.success("File verified! Processing batch...")
                    
                    # Align prediction vector order
                    eval_df = data[required_cols]
                    
                    with st.spinner("Analyzing transaction patterns..."):
                        start_time = time.time()
                        preds = model.predict(eval_df)
                        probs = model.predict_proba(eval_df)[:, 1]
                        elapsed = time.time() - start_time
                        
                    data['Prediction'] = np.where(preds == 1, '⚠️ FRAUD', '✅ SAFE')
                    data['Fraud_Risk_Probability'] = probs * 100
                    
                    # KPI Summary Cards
                    total_scans = len(data)
                    fraud_cases = int(np.sum(preds == 1))
                    safe_cases = int(np.sum(preds == 0))
                    fraud_ratio = (fraud_cases / total_scans) * 100
                    
                    col_tot, col_sf, col_fr, col_rt = st.columns(4)
                    
                    with col_tot:
                        st.markdown(f"""
                        <div class='glass-card' style='text-align:center;'>
                            <span class='metric-label'>Total Transactions</span>
                            <div class='metric-value'>{total_scans}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_sf:
                        st.markdown(f"""
                        <div class='glass-card' style='text-align:center;'>
                            <span class='metric-label'>Legit Cases</span>
                            <div class='metric-value' style='color:#34D399;'>{safe_cases}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_fr:
                        st.markdown(f"""
                        <div class='glass-card' style='text-align:center;'>
                            <span class='metric-label'>Flagged Fraud</span>
                            <div class='metric-value' style='color:#F87171;'>{fraud_cases}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_rt:
                        st.markdown(f"""
                        <div class='glass-card' style='text-align:center;'>
                            <span class='metric-label'>Anomaly Ratio</span>
                            <div class='metric-value'>{fraud_ratio:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Highlight fraud rows in UI table
                    st.markdown("#### 📝 Scan Summary Report")
                    
                    # Helper styling function
                    def style_rows(row):
                        if row['Prediction'] == '⚠️ FRAUD':
                            return ['background-color: rgba(239, 68, 68, 0.15); color: #F87171'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = data.style.apply(style_rows, axis=1)
                    st.dataframe(styled_df, height=350, use_container_width=True)
                    
                    # Download predicted results button
                    result_csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Complete Prediction Report (CSV)",
                        data=result_csv,
                        file_name="creditcard_scan_results.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error reading transaction CSV file: {e}")
