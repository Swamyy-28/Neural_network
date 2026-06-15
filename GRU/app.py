import streamlit as st
import numpy as np

# --- Page Setup ---
st.set_page_config(
    page_title="GRU Dashboard",
    page_icon="⚡",
    layout="wide"
)

# --- Custom CSS for Glassmorphic Light Tech UI ---
st.markdown("""
<style>
    /* Light Tech Canvas Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Frosted Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
        margin-bottom: 20px;
    }
    
    /* Modern Section Headers */
    .section-title {
        color: #1e293b;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-top: 0px;
        margin-bottom: 15px;
        border-left: 4px solid #4f46e5;
        padding-left: 10px;
    }
    
    /* Gate Badges (Minimal Neon Tech) */
    .gate-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .gate-label {
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .val-display {
        font-family: 'SF Mono', Consolas, monospace;
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Metric Context Colors */
    .lbl-reset { color: #ea580c; }
    .lbl-update { color: #2563eb; }
    .lbl-state { color: #16a34a; }
</style>
""", unsafe_allow_html=True)

# --- Clean Minimalist Header ---
st.markdown("<h1 style='text-align: center; color: #1e293b; font-weight: 800;'>⚡ GRU Gate Visualizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 15px; margin-bottom: 30px;'>A clean, mathematical look into recurrent neural network gating mechanisms.</p>", unsafe_allow_html=True)

# --- Simulation Engine ---
def run_simple_gru(text):
    tokens = [t.strip() for t in text.split() if t.strip()]
    if not tokens:
        return None, []
    
    h_t = np.zeros(2)
    history = []
    
    for idx, token in enumerate(tokens):
        seed = sum(ord(c) for c in token) % 100
        np.random.seed(seed)
        
        reset_gate = np.random.uniform(0.1, 0.9, 2)
        update_gate = np.random.uniform(0.1, 0.9, 2)
        
        clean_token = token.lower().strip(".,!?\"'")
        bias = 0.0
        if clean_token in ['fantastic', 'incredible', 'great', 'love', 'good']:
            bias = 0.6
        elif clean_token in ['bad', 'terrible', 'worst', 'boring', 'poor']:
            bias = -0.6
            
        candidate = np.tanh(reset_gate * h_t + np.random.normal(bias, 0.1, 2))
        h_t = (1 - update_gate) * h_t + update_gate * candidate
        h_t = np.clip(h_t, -1.0, 1.0)
        
        history.append({
            "step": idx + 1,
            "token": token,
            "reset": np.round(reset_gate, 2),
            "update": np.round(update_gate, 2),
            "state": np.round(h_t, 2)
        })
        
    final_score = 1 / (1 + np.exp(-np.mean(h_t) * 3))
    return final_score, history

# --- Layout Columns ---
col_left, col_right = st.columns([11, 10], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Sequence Configuration</h3>', unsafe_allow_html=True)
    user_input = st.text_input(
        "Text String to Vectorize:",
        value="The story structure was great, but the pacing felt bad."
    )
    st.markdown('<p style="color: #64748b; font-size: 13px; margin-top: 10px;">The recurrent architecture loops through your sentence string space-by-space to continuously adjust hidden weight properties.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Network Output Prediction</h3>', unsafe_allow_html=True)
    
    if user_input.strip():
        score, logs = run_simple_gru(user_input)
        
        if score >= 0.5:
            alert_box = f"""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 10px; text-align: center;">
                <span style="color: #16a34a; font-weight: 700; font-size: 16px;">📈 POSITIVE PREDICTION DETECTED</span>
                <p style="color: #166534; margin: 5px 0 0 0; font-size: 14px;">Calculated Sentiment Sigmoid: <b>{score:.1%}</b></p>
            </div>
            """
        else:
            alert_box = f"""
            <div style="background-color: #fef2f2; border: 1px solid #fecaca; padding: 15px; border-radius: 10px; text-align: center;">
                <span style="color: #dc2626; font-weight: 700; font-size: 16px;">📉 NEGATIVE PREDICTION DETECTED</span>
                <p style="color: #991b1b; margin: 5px 0 0 0; font-size: 14px;">Calculated Sentiment Sigmoid: <b>{(1-score):.1%}</b></p>
            </div>
            """
        st.markdown(alert_box, unsafe_allow_html=True)
    else:
        st.warning("Awaiting string matrix input data...")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Step-by-Step Breakdown ---
if user_input.strip() and 'logs' in locals():
    st.markdown('<h3 class="section-title" style="margin-top: 20px;">Execution Trace Logs</h3>', unsafe_allow_html=True)
    
    for step in logs:
        expander_label = f"📍 Node {step['step']} ➔ Processing Token: '{step['token']}'"
        with st.expander(expander_label):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown(f"""
                <div class="gate-container">
                    <div class="gate-label lbl-reset">Reset Gate ($r_t$)</div>
                    <div class="val-display">{step['reset']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown(f"""
                <div class="gate-container">
                    <div class="gate-label lbl-update">Update Gate ($z_t$)</div>
                    <div class="val-display">{step['update']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c3:
                st.markdown(f"""
                <div class="gate-container">
                    <div class="gate-label lbl-state">Hidden State Vector ($h_t$)</div>
                    <div class="val-display">{step['state']}</div>
                </div>
                """, unsafe_allow_html=True)


