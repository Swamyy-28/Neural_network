import streamlit as st
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="LSTM Sentiment Analysis Engine",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Clean UI CSS Injection ---
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
    
    /* 4. Text Area Overrides */
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

    /* 6. Component Pill Badges */
    .lstm-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.75rem;
        margin-right: 5px;
        text-transform: uppercase;
    }
    .cell-state { background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
    .hidden-state { background-color: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }
</style>
""", unsafe_allow_html=True)

# --- Mock LSTM Recurrent Core Engine ---
def simulate_lstm_sentiment(text):
    """Simulates LSTM sequence accumulation with Cell States & Hidden States."""
    words = [w.strip() for w in text.split() if w.strip()]
    if not words:
        return 0.5, 0, []
    
    # LSTM maintains two vectors: Cell State (C_t) and Hidden State (H_t)
    c_t = np.zeros(3)
    h_t = np.zeros(3)
    word_traces = []
    
    for idx, word in enumerate(words):
        seed = sum(ord(char) for char in word) % 100
        np.random.seed(seed)
        
        # Define basic sentiment impacts per token
        clean_word = word.lower().strip(".,!?\"'")
        sentiment_pull = 0.0
        if clean_word in ['amazing', 'good', 'love', 'masterpiece', 'great', 'excellent', 'beautiful']:
            sentiment_pull = 0.5
        elif clean_word in ['bad', 'boring', 'terrible', 'worst', 'waste', 'horrible']:
            sentiment_pull = -0.5
            
        # Simulating LSTM Gates: Forget, Input, Output
        forget_gate = np.random.uniform(0.6, 0.95, 3) # Long memory retention
        input_gate = np.random.uniform(0.2, 0.8, 3)
        output_gate = np.random.uniform(0.3, 0.8, 3)
        
        # Update Cell State and Hidden State formulas
        candidate_cell = np.tanh(np.random.normal(sentiment_pull, 0.2, 3))
        c_t = (forget_gate * c_t) + (input_gate * candidate_cell)
        h_t = output_gate * np.tanh(c_t)
        
        word_traces.append({
            "index": idx + 1,
            "word": word,
            "cell_state": c_t.copy(),
            "hidden_state": h_t.copy()
        })
        
    # Translate final state vector average into a classification probability
    final_score = 1 / (1 + np.exp(-np.mean(h_t) * 4))
    return final_score, len(words), word_traces

# --- Centered Header Panel ---
st.markdown("""
<div class="title-container">
    <div class="main-title">🎭 LSTM Sentiment Analysis Engine</div>
    <div class="sub-title">Recurrent Neural Network Model for Sequential NLP Classification</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Sidebar Parameter Selection ---
st.sidebar.markdown("### 🔧 Model Architecture")
vocab_size = st.sidebar.selectbox("Vocabulary Embedding Size", ["10,000 tokens", "50,000 tokens"])
lstm_units = st.sidebar.slider("LSTM Hidden Units Layers", 32, 256, 128)
dropout_rate = st.sidebar.slider("Spatial Dropout Rate", 0.0, 0.5, 0.2)

# --- UI Multi-Column Content Frame ---
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
    st.subheader("📥 Input Target Text")
    user_input = st.text_area(
        "Enter sequence to analyze sentiment payload:",
        value="The cinematic score was an amazing masterpiece, but some parts felt a bit boring.",
        height=120
    )
    run_analysis = st.button("🚀 Analyze Sentiment Sequence", use_container_width=True)

with col_right:
    st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
    st.subheader("📊 Output Inference Dashboard")
    
    if run_analysis and user_input.strip():
        with st.spinner("Propagating inputs through recurrent LSTM cells..."):
            score, token_count, execution_history = simulate_lstm_sentiment(user_input)
            
            # Simulated progress step matching premium standard
            progress = st.progress(0)
            for step in range(100):
                time.sleep(0.002)
                progress.progress(step + 1)
                
            # Render Clean KPIs
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Token Count", f"{token_count} words")
            kpi2.metric("LSTM Layer", f"{lstm_units} Units")
            kpi3.metric("Retention Rate", f"{(1 - dropout_rate):.0%}")
            
            st.divider()
            
            # Binary Sentiment Output Block
            st.markdown("#### **Classification Inference**")
            if score >= 0.5:
                st.success(f"🟢 **POSITIVE SENTIMENT** Confidence Level: {score:.2%}")
            else:
                st.error(f"🔴 **NEGATIVE SENTIMENT** Confidence Level: {(1 - score):.2%}")
                
            st.divider()
            
            # Final Vectors Visualization
            if execution_history:
                last_step = execution_history[-1]
                st.markdown("#### **Terminal Memory Arrays**")
                
                st.markdown("<span class='lstm-badge cell-state'>Cell State Vector ($C_t$)</span> *Long Term Memory*", unsafe_allow_html=True)
                st.code(f"{last_step['cell_state']}")
                
                st.markdown("<span class='lstm-badge hidden-state'>Hidden State Vector ($H_t$)</span> *Current Word Context*", unsafe_allow_html=True)
                st.code(f"{last_step['hidden_state']}")
                
    else:
        st.info("💡 Input a text string on the left panel and hit analyze to observe cell gate operations.")

# --- Full Width Educational Framework Footprint ---
st.markdown('<div class="custom-block"></div>', unsafe_allow_html=True)
st.subheader("📐 Inside the Long Short-Term Memory Architecture")
st.write("""
Standard Recurrent Networks suffer from **vanishing gradients** over long texts. An **LSTM layer** overrides this by keeping 
a continuous internal highway called the **Cell State ($C_t$)**. At every word, three distinct mathematical gates filter information:
""")

math_col1, math_col2, math_col3 = st.columns(3)
with math_col1:
    st.markdown("**1. Forget Gate ($f_t$)**")
    st.caption("Decides what old context matrices to erase.")
with math_col2:
    st.markdown("**2. Input Gate ($i_t$)**")
    st.caption("Determines what new data payload to store.")
with math_col3:
    st.markdown("**3. Output Gate ($o_t$)**")
    st.caption("Funnels the updated metrics into the next Hidden State ($h_t$).")

# --- Step-by-Step Step Log Grid ---
if run_analysis and user_input.strip() and 'execution_history' in locals():
    st.subheader("📋 Step-by-Step Sequence Trace")
    st.write("Expand individual token indices to view the cell update timeline:")
    
    for log_item in execution_history:
        with st.expander(label=f"Token Block {log_item['index']}: '{log_item['word']}'"):
            trace_col1, trace_col2 = st.columns(2)
            trace_col1.markdown("**Cell State Vector Accumulation:**")
            trace_col1.code(f"{log_item['cell_state']}")
            trace_col2.markdown("**Resulting Hidden Vector Output:**")
            trace_col2.code(f"{log_item['hidden_state']}")