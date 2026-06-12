import streamlit as st
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="BERT Contextual Encoder Workbench",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Clean Light-Theme UI CSS Injection ---
st.markdown("""
<style>
    /* 1. Page Background Setup */
    .stApp {
        background: linear-gradient(180deg, #f5f3ff 0%, #f8fafc 100%) !important;
        color: #0f172a !important;
    }
    
    /* 2. Turn Streamlit native blocks into clean white shadow cards */
    [data-testid="stVerticalBlock"] > div:has(div.custom-block),
    .stBlock {
        background-color: #ffffff !important;
        border-radius: 16px !important;
        padding: 28px !important;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.05), 0 8px 16px -6px rgba(0, 0, 0, 0.03) !important;
        border: 2px solid #e9d5ff !important;
        margin-bottom: 25px !important;
    }
    
    /* 3. Gradient Accent Header Panel */
    .title-container {
        text-align: center;
        padding: 24px 0;
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(124, 58, 237, 0.15);
    }
    .main-title {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 2.6rem !important;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }
    .sub-title {
        color: #ddd6fe !important;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 5px;
    }
    
    /* 4. Inputs Text Area Overrides */
    .stTextArea textarea {
        background-color: #fcfbff !important;
        border: 2px solid #ddd6fe !important;
        color: #1e1b4b !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }
    
    /* 5. Custom Button Layout */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(124, 58, 237, 0.3) !important;
    }

    /* 6. BERT Component Badges */
    .bert-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.75rem;
        margin-right: 5px;
        text-transform: uppercase;
    }
    .token-id { background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
    .attention-head { background-color: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }
</style>
""", unsafe_allow_html=True)

# --- Simulation Engine for BERT Mechanism ---
def simulate_bert_encoder(text):
    """Simulates BERT tokenization, special tags injection, and attention weighting."""
    raw_words = [w.strip() for w in text.split() if w.strip()]
    if not raw_words:
        return [], 0.5
    
    # BERT explicitly prepends a [CLS] classification token and appends a [SEP] separator token
    tokens = ["[CLS]"] + raw_words + ["[SEP]"]
    num_tokens = len(tokens)
    
    # Generate mock Transformer attention weights matrix (scaled dot-product attention)
    np.random.seed(len(text))
    attention_matrix = np.random.uniform(0.05, 0.2, (num_tokens, num_tokens))
    
    # Highlight high-importance semantic links (e.g., matching negations to targets)
    lower_tokens = [t.lower().strip(".,!?\"'") for t in tokens]
    if "not" in lower_tokens:
        not_idx = lower_tokens.index("not")
        for i, tok in enumerate(lower_tokens):
            if tok in ["boring", "bad", "terrible", "excellent", "amazing"]:
                attention_matrix[not_idx, i] = 0.75
                attention_matrix[i, not_idx] = 0.65
                
    # Softmax normalize row values to represent legal probabilities summing to 1.0
    for i in range(num_tokens):
        exp_row = np.exp(attention_matrix[i] * 3)
        attention_matrix[i] = exp_row / np.sum(exp_row)
        
    # Calculate a final classification evaluation based on the [CLS] attention vector
    cls_vector = attention_matrix[0]
    positive_signals = ["amazing", "masterpiece", "excellent", "perfect", "good"]
    negative_signals = ["boring", "bad", "terrible", "waste", "poor"]
    
    base_sentiment = 0.5
    for idx, tok in enumerate(lower_tokens):
        if tok in positive_signals:
            base_sentiment += (cls_vector[idx] * 1.5)
        elif tok in negative_signals:
            base_sentiment -= (cls_vector[idx] * 1.5)
            
    if "not" in lower_tokens:
        # Flip sentiment if basic negation structures are linked
        base_sentiment = 1.0 - base_sentiment if base_sentiment < 0.5 else base_sentiment + 0.1

    final_score = np.clip(base_sentiment, 0.02, 0.98)
    return tokens, attention_matrix, final_score

# --- Centered Premium Header Panel ---
st.markdown("""
<div class="title-container">
    <div class="main-title">🤖 BERT Transformer Encoder Workspace</div>
    <div class="sub-title">Bidirectional Encoder Representations from Transformers — Attention Matrix Layer Analysis</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Sidebar Configuration Defaults ---
st.sidebar.markdown("### 🔧 Transformer Settings")
bert_variant = st.sidebar.selectbox("Model Core Type", ["bert-base-uncased (12-Layers)", "bert-large-uncased (24-Layers)"])
num_heads = st.sidebar.slider("Active Attention Heads", 4, 16, 12)

# --- Dual Column Layout Grid System ---
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
    st.subheader("📥 Input Target Sequence")
    sample_text = st.text_area(
        "Enter sentence to execute through transformer layers:",
        value="Not a boring film, it was an amazing masterpiece with excellent pacing.",
        height=110
    )
    trigger_process = st.button("🚀 Run Transformer Tokenizer & Encoder", use_container_width=True)

with col_right:
    st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
    st.subheader("📊 Output Metrics Engine")
    
    if trigger_process and sample_text.strip():
        with st.spinner("Tokenizing and executing multi-head attention arrays..."):
            tokens, attention_weights, sentiment_score = simulate_bert_encoder(sample_text)
            
            progress = st.progress(0)
            for pct in range(100):
                time.sleep(0.001)
                progress.progress(pct + 1)
            
            # Numeric Feature Metrics Row matching your screenshot style perfectly
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("BERT Sub-tokens", len(tokens))
            m_col2.metric("Attention Matrix", f"{len(tokens)} × {len(tokens)}")
            m_col3.metric("Attention Heads", f"{num_heads} Heads")
            
            st.divider()
            
            # High Contrast Evaluation Callouts
            st.markdown("#### **Classification Inference (Via [CLS] Token Key)**")
            if sentiment_score >= 0.5:
                st.success(f"🟢 **POSITIVE SENTIMENT** Confidence Level: {sentiment_score:.2%}")
            else:
                st.error(f"🔴 **NEGATIVE SENTIMENT** Confidence Level: {(1 - sentiment_score):.2%}")
                
            st.divider()
            
            # Displaying Token Payload Arrays
            st.markdown("#### **BERT Token Embedding Map**")
            st.markdown("<span class='bert-badge token-id'>Processed Input Tensors</span>", unsafe_allow_html=True)
            st.code(f"{tokens}")
    else:
        st.info("💡 Write an input string sequence inside the field on the left and click execute to trigger attention mapping.")


# --- Attention Matrix Weight Map Breakdown ---
if trigger_process and sample_text.strip() and 'tokens' in locals():
    st.subheader("📋 Contextual Attention Maps Matrix")
    st.write("Expand each token block below to see its statistical attention distribution weight profile across the rest of the sentence:")
    
    for idx, token in enumerate(tokens):
        with st.expander(label=f"Token Block Index {idx} ➔ '{token}' Attention Allocation"):
            st.markdown(f"<span class='bert-badge attention-head'>Attention Vector Row values for '{token}'</span>", unsafe_allow_html=True)
            
            # Map attention weights directly back to target words for high readability
            token_weight_mapping = {tokens[i]: f"{attention_weights[idx][i]:.4f}" for i in range(len(tokens))}
            st.json(token_weight_mapping)
