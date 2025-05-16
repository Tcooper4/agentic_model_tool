import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
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
st.title("🚀 Advanced Agentic Model Creation Tool (Parallelized Optimization + LLM Selector)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request or Prompt (e.g., 'Predict AAPL')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)
auto_reoptimize = st.sidebar.checkbox("🔄 Auto Re-Optimize Models Every Hour", value=True)
reoptimize_interval = st.sidebar.number_input("Re-Optimization Interval (Seconds)", min_value=60, value=3600)
custom_strategy = st.sidebar.checkbox("📝 Enable Custom Strategy Creation", value=True)

# LLM Selector
llm_type = st.sidebar.selectbox("🔑 Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])
if llm_type == "OpenAI (GPT-4)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Required for GPT-4)", type="password")

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
    def calculate_rsi(df, period=14):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def add_technical_indicators(df):
        df['RSI'] = calculate_rsi(df)
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
        return df

    data = add_technical_indicators(data)

    # Display Historical Data Only (No Future Dates)
    st.subheader("📊 Historical Data Visualization")
    st.line_chart(data[['Date', 'Close']].set_index('Date'))

    # Forecasting Section (Separate from Historical Data)
    st.subheader("📈 Forecasting with Prophet (Future Dates)")
    model = Prophet()
    prophet_df = pd.DataFrame({'ds': data['Date'], 'y': data['Close']})
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Data'))
    forecast_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    st.plotly_chart(forecast_fig)

    # Display Forecast Data
    st.subheader("🔮 Forecast Data")
    st.write(forecast[['ds', 'yhat']].tail(forecast_period))

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
