import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ANN",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# PREMIUM MEDICAL LIGHT UI CSS THEME
# =====================================================
st.markdown("""
    <style>
    /* Global App Canvas */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Elegant Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Primary Gradient Header styling */
    .main-title {
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Minimalist Elegant Crisp Cards */
    .medical-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .medical-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    
    /* Metrics Flex Boxes */
    .metric-box {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-lbl { color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .metric-val { color: #1e3a8a; font-size: 2rem; font-weight: 700; }

    /* Custom Input Labels Adjustment */
    label[data-testid="stWidgetLabel"] p {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    
    hr { border-color: #e2e8f0 !important; }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATASET
# =====================================================
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# =====================================================
# PREPROCESSING
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# ANN MODEL
# =====================================================
@st.cache_resource
def train_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(30,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        epochs=30, batch_size=16, verbose=0
    )
    return model

model = train_model()

# =====================================================
# ACCURACY CALCULATIONS
# =====================================================
predictions = model.predict(X_test, verbose=0)
predictions = (predictions > 0.5).astype(int)
accuracy = accuracy_score(y_test, predictions)

# =====================================================
# TITLE & HERO HEADLINE
# =====================================================
st.markdown('<h1 class="title">ANN Breast Cancer predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Clinical Diagnostic Artificial Neural Network Engine trained on the Breast Cancer Wisconsin Matrix Data Profile.</p>', unsafe_allow_html=True)

# Layout Splitting: Diagnostic Panel vs Target Specifications
col_left, col_right = st.columns([4, 8], gap="large")

with col_left:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.subheader("📊 Engine Performance")
    
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Validated Accuracy</div>
        <div class="metric-val">{accuracy*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.subheader("📋 Classification Key Reference")
    info = pd.DataFrame({
        "Target Value": [0, 1],
        "Clinical Pathology Category": ["Malignant (High-Risk Case)", "Benign (Non-Cancerous Variant)"]
    })
    st.dataframe(info, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# USER INPUT FEEDS MAP
# =====================================================
st.markdown('<h3>🩺 Feature Variables Entry Desk</h3>', unsafe_allow_html=True)
st.markdown('<div class="medical-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
inputs = []

for i, feature in enumerate(feature_names[:10]):
    cleaned_label = feature.replace(" ", "_").title()
    if i % 2 == 0:
        with col1:
            val = st.number_input(
                f"🧬 {cleaned_label}",
                value=float(X[:, i].mean()),
                key=f"feat_{i}"
            )
    else:
        with col2:
            val = st.number_input(
                f"🧬 {cleaned_label}",
                value=float(X[:, i].mean()),
                key=f"feat_{i}"
            )
    inputs.append(val)

# Remaining feature structures auto-filled using standard dataset feature means
for i in range(10, 30):
    inputs.append(float(X[:, i].mean()))

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PREDICT INTERACTION BLOCK
# =====================================================
if st.button("Run Diagnostic Classifier Pipeline", type="primary", use_container_width=True):
    
    sample = np.array(inputs).reshape(1, -1)
    sample = scaler.transform(sample)
    
    prediction = model.predict(sample, verbose=0)
    probability = float(prediction[0][0])
    
    st.markdown("---")
    
    col_out1, col_out2 = st.columns([5, 7], gap="large")
    
    with col_out1:
        st.markdown('<div class="medical-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("📌 Classification Assessment Result")
        
        if probability >= 0.5:
            st.success("🟢 Diagnosis Verdict: Benign Variant Detected")
        else:
            st.error("🔴 Diagnosis Verdict: Malignant Tumor Profile Identified")
            
        st.info(f"🧬 Operational Confidence Index: {max(probability, 1-probability)*100:.2f}%")
        
        st.markdown("<br><b>Metric Probability Breakdown Matrix:</b>", unsafe_allow_html=True)
        probs = [1 - probability, probability]
        labels = ["Malignant", "Benign"]
        prob_df = pd.DataFrame({"Class Label Category": labels, "Calculated Probability Field": probs})
        st.dataframe(prob_df.sort_values(by="Calculated Probability Field", ascending=False), hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_out2:
        st.markdown('<div class="medical-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("📈 Spatial Diagnostic Probability Chart")
        
        # Interactive Plotly chart to replace old matplotlib rendering engine layout 
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=labels, y=probs,
            marker_color=['#ef4444', '#10b981'],
            width=0.4
        ))
        fig_bars.update_layout(
            yaxis=dict(title="Probability Matrix", range=[0, 1], gridcolor='#f1f5f9'),
            xaxis=dict(title="Pathological Condition Category"),
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bars, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# DATASET PREVIEW FOOTER
# =====================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="medical-card">', unsafe_allow_html=True)
st.subheader("📁 Primary Structural Training Log Frameworks")
df = pd.DataFrame(X, columns=feature_names)
st.dataframe(df.head(5), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)