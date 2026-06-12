import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# --- 1. PAGE SETUP & DESIGN LAYOUT ---
st.set_page_config(
    page_title="PulseLSTM | Financial Analytics Core",
    page_icon="📈",
    layout="wide"
)

# --- 2. PREMIUM MINT & SLATE MEDICAL/FINANCIAL LIGHT CSS ---
st.markdown("""
    <style>
    /* Global App Framework Elements */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Clean Crisp Sidebar Structuring */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Elegant Title Typography Setup */
    .main-title {
        background: linear-gradient(135deg, #0f172a 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Minimalist Micro Cards for Visual Elements */
    .finance-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .finance-card:hover {
        border-color: #10b981;
        transform: translateY(-2px);
    }
    
    /* Custom Indicator Metrics Blocks */
    .metric-box {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-lbl { color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .metric-val { color: #0f172a; font-size: 2rem; font-weight: 700; }

    hr { border-color: #e2e8f0 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. MAIN HEADER BLOCK ---
st.markdown('<h1 class="main-title">PulseLSTM Financial Core</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Recurrent Neural Network sandbox engine utilizing Long Short-Term Memory nodes for sequential historical data evaluation.</p>', unsafe_allow_html=True)

# --- 4. CONTROL INTERFACE SIDEBAR PANEL ---
st.sidebar.markdown("## ⚙️ Model Framework Controls")
st.sidebar.markdown("---")

st.sidebar.subheader("📈 Target Equity Profile")
ticker_symbol = st.sidebar.text_input("YFinance Ticker Token", value="AAPL")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Recurrent LSTM Architecture Layout")
lstm_units_1 = st.sidebar.slider("LSTM Neurons Layer A", 20, 100, 50, step=10)
lstm_units_2 = st.sidebar.slider("LSTM Neurons Layer B", 10, 50, 30, step=10)
dropout_rate = st.sidebar.slider("Dropout Regularization Value", 0.0, 0.4, 0.2, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Optimization Constants")
training_epochs = st.sidebar.slider("Iteration Execution Count (Epochs)", 5, 50, 15, step=5)
batch_size_value = st.sidebar.selectbox("Batch Computation Window Size", options=[16, 32, 64], index=1)

# --- 5. STREAM PIPELINE TIME-SERIES DATA LOADING ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_sequential_market_logs(ticker):
    """Downloads historical pricing profiles from public global market records."""
    df = yf.download(ticker, start="2020-01-01", end="2026-01-01")
    return df

with st.spinner("Downloading sequential time-series vector sets..."):
    try:
        raw_data = fetch_sequential_market_logs(ticker_symbol)
        if raw_data.empty:
            st.error("Error: Received completely empty token payload check from source records.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to fetch market metrics for code symbol {ticker_symbol}: {e}")
        st.stop()

# Isolate Closing Data Matrices
close_dataset = raw_data[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_dataset)

# Establish Train-Test Sizing Splitting Configurations (80% Training)
training_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[0:int(training_len), :]

# Formulate Step Sliding Windows (60 past steps to predict 1 next step value)
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
# Reshape feature dimensions to fit expected 3D LSTM Tensor Shapes [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# --- 6. CORE TENSORFLOW LSTM MACHINE FRAMEWORK COMPILATION ---
@st.cache_resource
def compile_lstm_architecture(units_a, units_b, drop, epochs, batch_size, _X_train, _y_train):
    """Compiles and fits a Deep Recurrent Sequential Network model."""
    lstm_net = Sequential([
        LSTM(units_a, return_sequences=True, input_shape=(_X_train.shape[1], 1)),
        Dropout(drop),
        LSTM(units_b, return_sequences=False),
        Dropout(drop),
        Dense(25),
        Dense(1)
    ])
    
    lstm_net.compile(optimizer='adam', loss='mean_squared_error')
    lstm_net.fit(_X_train, _y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    return lstm_net

with st.spinner("Optimizing Recurrent Network Weights (Training LSTM)..."):
    trained_lstm = compile_lstm_architecture(
        lstm_units_1, lstm_units_2, dropout_rate, 
        training_epochs, batch_size_value, X_train, y_train
    )

# --- 7. EVALUATING FORWARD VALIDATION SET FORECASTS ---
test_data = scaled_data[training_len - 60:, :]
X_test = []
y_test = close_dataset[training_len:, :]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Compute Tensor Predictions and invert scaling layers to get raw currencies
model_predictions = trained_lstm.predict(X_test, verbose=0)
model_predictions = scaler.inverse_transform(model_predictions)

# Calculate Evaluation Variance Metrics (Root Mean Squared Error)
rmse_metric = np.sqrt(np.mean(((model_predictions - y_test) ** 2)))

# --- 8. RENDER SCREEN VISUAL ANALYTICS LAYOUTS ---
col_card_left, col_card_right = st.columns([4, 8], gap="large")

with col_card_left:
    st.markdown('<div class="finance-card">', unsafe_allow_html=True)
    st.subheader("📊 Network Diagnostic Scores")
    st.markdown(f"""
    <div class="metric-box" style="margin-bottom:1rem;">
        <div class="metric-lbl">Root Mean Squared Error (RMSE)</div>
        <div class="metric-val">$ {rmse_metric:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Lower RMSE signifies that the model's recurrent paths successfully mapped the cyclical time trends.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_card_right:
    st.markdown('<div class="finance-card">', unsafe_allow_html=True)
    st.subheader("📋 Underlying Historical Log Snippet")
    # Clean up column visual presentations for dataframe displays
    display_df = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(4)
    st.dataframe(display_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 9. RENDER DYNAMIC SEQUENTIAL TREND CHART ---
st.markdown('<h3>📉 Evaluated Target Prediction Validation Waves</h3>', unsafe_allow_html=True)
st.markdown('<div class="finance-card">', unsafe_allow_html=True)

# Parse time indices out of base pandas frame structures
train_df = raw_data.iloc[:training_len]
valid_df = raw_data.iloc[training_len:].copy()
valid_df['Predictions'] = model_predictions

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'].values.flatten(), mode='lines', name='Historical Baseline Set', line=dict(color='#64748b')))
fig_trend.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Close'].values.flatten(), mode='lines', name='Actual Future Target', line=dict(color='#0f172a', width=2)))
fig_trend.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Predictions'].values.flatten(), mode='lines', name='LSTM Computed Path', line=dict(color='#10b981', width=2, dash='dash')))

fig_trend.update_layout(
    xaxis=dict(title="Timeline Horizon", gridcolor='#e2e8f0'),
    yaxis=dict(title="Asset Close Price Valuation ($)", gridcolor='#e2e8f0'),
    height=450, margin=dict(l=15, r=15, t=15, b=15),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_trend, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)