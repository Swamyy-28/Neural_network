import streamlit as st
import requests
import json

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Universal Translator | EN ➔ TE",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- CSS UI STYLING -----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .title-container {
        text-align: center;
        padding: 25px 0px;
        margin-bottom: 20px;
        background: rgba(255, 255, 255, 0.4);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    .title-main {
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #1e3a8a !important;
        margin: 0;
    }
    .title-sub {
        font-size: 16px;
        color: #4b5563;
        margin-top: 5px;
    }
    div[data-testid="stTextHeight"] {
        border-radius: 12px !important;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.85);
        padding: 24px;
        border-radius: 16px;
        border-left: 8px solid #2563eb;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        min-height: 200px;
    }
    .result-header {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
        margin-bottom: 12px;
        font-weight: 600;
    }
    .result-text {
        font-size: 26px;
        color: #111827;
        font-weight: 500;
        line-height: 1.6;
    }
    .empty-state {
        color: #9ca3af;
        font-style: italic;
        padding-top: 40px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- RELIABLE GOOGLE TRANSLATE ENGINE -----------------

def translate_to_telugu(text):
    """
    Highly reliable translation engine using Google's direct RPC translate service endpoint.
    Handles slang, typos, names, and punctuation flawlessly without API tokens.
    """
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",      # Source Language: English
            "tl": "te",      # Target Language: Telugu
            "dt": "t",       # Data Type: Text
            "q": text
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            # The API returns a nested list structure. We parse out the text blocks.
            response_json = response.json()
            translated_chunks = [sentence[0] for sentence in response_json[0] if sentence[0]]
            translated_text = "".join(translated_chunks)
            return translated_text
        else:
            return "Error: Translation node failed to respond. Please try again."
            
    except Exception as e:
        return f"System Connection Error: Couldn't reach translation routing layer. ({str(e)})"

# ----------------- UI WINDOW RUNTIME -----------------

st.markdown("""
<div class="title-container">
    <h1 class="title-main">🌍 English ➔ Telugu Cloud Translator</h1>
    <div class="title-sub">High-fidelity translation matrix with dynamic typography layers</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📝 Source Document")
    text = st.text_area(
        label="English Input Layer",
        label_visibility="collapsed",
        height=220,
        placeholder="Type your English phrases or sentences here..."
    )
    
    translate_btn = st.button(
        "⚡ Transliterate Sequence",
        use_container_width=True,
        type="primary"
    )

with col2:
    st.markdown("### 🎯 Target Result")
    
    if translate_btn:
        if not text.strip():
            st.toast("⚠️ Input sequence can't be blank!", icon="❌")
            st.markdown(
                '<div class="result-card"><p class="empty-state">Waiting for valid text input...</p></div>', 
                unsafe_allow_html=True
            )
        else:
            with st.spinner("Compiling translation arrays..."):
                translated_output = translate_to_telugu(text)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">Telugu Translation</div>
                <div class="result-text">{translated_output}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card">
            <p class="empty-state">Enter an English statement on the left and trigger the compilation to fetch output.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><hr><center style='color:#6b7280; font-size:12px;'>Optimized Engine Framework Layer</center>", unsafe_allow_html=True)