import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from prophet import Prophet
import yfinance as yf
import plotly.graph_objects as go
import json
import os
import pickle

st.set_page_config(page_title="🚀 Advanced Agentic Model Creation Tool", layout="wide")
st.title("🚀 Advanced Agentic Model Creation Tool (Vectorized Indicators + Fast Strategy Optimization)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request or Prompt (e.g., 'Predict AAPL')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)
auto_reoptimize = st.sidebar.checkbox("🔄 Auto Re-Optimize Models Every Hour", value=True)
reoptimize_interval = st.sidebar.number_input("Re-Optimization Interval (Seconds)", min_value=60, value=3600)
custom_strategy = st.sidebar.checkbox("📝 Enable Custom Strategy Creation", value=True)

# Directory for Saving Models and Strategies
model_dir = "models/"
strategy_file = "optimized_strategies.json"
os.makedirs(model_dir, exist_ok=True)

# Smart Data Sourcing (Real-Time)
@st.cache_data(ttl=60 * 60)
def fetch_data(ticker: str):
    try:
        data = yf.download(ticker, period="2y", interval="1d")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None

# Data Source Detection
def detect_data_source(prompt):
    prompt = prompt.lower()
    if "sp500" in prompt:
        return "SPY"
    words = prompt.split()
    for word in words:
        if word.isalpha() and len(word) <= 5:
            return word.upper()
    return None

ticker = detect_data_source(prompt)
data = fetch_data(ticker) if ticker else None

if data is not None and not data.empty:
    st.write(f"✅ Data Loaded for {ticker}")
    st.write(data.head())

    # Optimized Technical Indicators (Fully Vectorized)
    def add_technical_indicators(df):
        # RSI Calculation (Fully Vectorized)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD Calculation (Fully Vectorized)
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()

        # SMA and EMA (Fully Vectorized)
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        return df

    data = add_technical_indicators(data)

    # Load and Save Optimized Strategies
    def save_optimized_strategies(strategies):
        with open(strategy_file, "w") as file:
            json.dump(strategies, file)

    def load_optimized_strategies():
        if os.path.exists(strategy_file):
            with open(strategy_file, "r") as file:
                return json.load(file)
        return {}

    optimized_strategies = load_optimized_strategies()

    # Adaptive Strategy Optimization (Fast + Vectorized)
    def optimize_strategy(df):
        st.subheader("🔧 Adaptive Strategy Optimization")
        strategies = {}

        # RSI Optimization
        best_rsi_return = -np.inf
        best_rsi_period = 14
        for period in range(5, 50):
            rsi = calculate_rsi(df, period)
            df['RSI_Signal'] = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))
            strategy_return = (df['RSI_Signal'].shift(1) * df['Close'].pct_change()).cumsum().iloc[-1]
            if strategy_return > best_rsi_return:
                best_rsi_return = strategy_return
                best_rsi_period = period

        strategies['RSI_Optimized'] = best_rsi_period
        st.write(f"✅ Optimized RSI Period: {best_rsi_period}")

        # MACD Optimization
        best_macd_return = -np.inf
        best_fast, best_slow = 12, 26
        for fast in range(5, 30):
            for slow in range(fast + 1, 50):
                macd = df['Close'].ewm(span=fast, adjust=False).mean() - df['Close'].ewm(span=slow, adjust=False).mean()
                df['MACD_Signal'] = np.where(macd > 0, 1, -1)
                strategy_return = (df['MACD_Signal'].shift(1) * df['Close'].pct_change()).cumsum().iloc[-1]
                if strategy_return > best_macd_return:
                    best_macd_return = strategy_return
                    best_fast, best_slow = fast, slow

        strategies['MACD_Optimized'] = (best_fast, best_slow)
        st.write(f"✅ Optimized MACD: Fast={best_fast}, Slow={best_slow}")

        save_optimized_strategies(strategies)
        return strategies

    # Adaptive Strategy Management
    if custom_strategy:
        st.subheader("📝 Custom Strategy Creation")
        if st.button("Optimize Strategies"):
            optimized_strategies = optimize_strategy(data)
            st.write("✅ Strategies Optimized:")
            st.write(optimized_strategies)
        else:
            if optimized_strategies:
                st.write("✅ Loaded Optimized Strategies:")
                st.write(optimized_strategies)
            else:
                st.write("❌ No optimized strategies found.")

    # Adaptive Strategy Backtesting
    st.subheader("📊 Adaptive Strategy Backtesting")
    if optimized_strategies:
        data = add_technical_indicators(data)
        data['RSI_Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
        data['MACD_Signal'] = np.where(data['MACD'] > 0, 1, -1)
        data['Strategy_Return'] = data['RSI_Signal'].shift(1) * data['Close'].pct_change() + data['MACD_Signal'].shift(1) * data['Close'].pct_change()
        strategy_performance = data['Strategy_Return'].cumsum()
        st.write(f"✅ Adaptive Strategy Return: {strategy_performance.iloc[-1]:.4f}")
        st.line_chart(strategy_performance)
else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
