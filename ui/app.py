import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet
import yfinance as yf
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="🚀 Fully Agentic Model Creation Tool", layout="wide")
st.title("🚀 Fully Agentic Model Creation Tool (Adaptive, Real-Time, Self-Optimizing)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request (e.g., 'Predict SP500', 'Optimize RSI Strategy')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)
custom_strategy = st.sidebar.checkbox("📝 Enable Custom Strategy Creation", value=True)

# Agentic Data Loading and Column Detection
@st.cache_data(ttl=60 * 60)
def agentic_data_sourcing(prompt):
    prompt = prompt.lower()
    ticker = None
    
    if "sp500" in prompt:
        ticker = "SPY"
    elif "stock" in prompt or "etf" in prompt:
        words = prompt.split()
        for word in words:
            if word.isalpha() and len(word) <= 5:
                ticker = word.upper()
                break
    
    if not ticker:
        st.error("❌ Unable to detect ticker. Please enter a valid prompt.")
        return None
    
    st.write(f"✅ Fetching Real-Time Data for {ticker} (Yahoo Finance)")
    data = yf.download(ticker, period="2y", interval="1d")
    if data.empty:
        st.error(f"❌ No data found for {ticker}. Please enter a valid ticker.")
        return None
    
    data.reset_index(inplace=True)
    data.columns = [str(col) for col in data.columns]

    # Detecting the best column for analysis
    target_columns = ['Close', 'Adj Close', 'Last', 'Price']
    target_column = next((col for col in data.columns if any(key.lower() in col.lower() for key in target_columns)), None)

    if not target_column:
        st.error("❌ Unable to identify a suitable column for analysis.")
        return None
    
    st.write(f"✅ Using '{target_column}' column for analysis.")
    data.rename(columns={target_column: "Target"}, inplace=True)
    return data

# Load Data Based on User Prompt
data = agentic_data_sourcing(prompt)

if data is not None and not data.empty:
    st.write("✅ Real-Time Data Loaded")
    st.write(data.head())

    # Adaptive Model and Strategy Recognition
    def agentic_model_and_strategy(prompt, data):
        if "forecast" in prompt:
            st.subheader("📈 Adaptive Forecasting")
            y = data['Target'].values
            best_model, best_forecast = optimize_forecasting_models(y)
            st.write(f"✅ Best Forecasting Model: {best_model}")
            display_forecast(data, y, best_forecast)

        if "rsi" in prompt:
            optimize_and_display_strategy(data, "RSI")
        if "macd" in prompt:
            optimize_and_display_strategy(data, "MACD")
        if "sma" in prompt:
            optimize_and_display_strategy(data, "SMA")
        
        if custom_strategy:
            st.subheader("📝 Custom Strategy Creation")
            custom_signal = st.text_input("Enter Custom Strategy Logic (e.g., 'RSI > 70 and MACD > 0')")
            if custom_signal:
                try:
                    data['Custom_Signal'] = data.eval(custom_signal).astype(int)
                    data['Custom_Strategy_Return'] = data['Custom_Signal'].shift(1) * data['Target'].pct_change()
                    st.write(f"✅ Custom Strategy Performance: {data['Custom_Strategy_Return'].cumsum().iloc[-1]:.4f}")
                except:
                    st.error("❌ Invalid custom strategy. Please check your syntax.")

    def optimize_forecasting_models(y):
        arima_model = ARIMA(y, order=(5, 1, 0)).fit()
        arima_forecast = arima_model.forecast(steps=forecast_period)
        
        model = Prophet()
        prophet_df = pd.DataFrame({'ds': data['Date'], 'y': data['Target']})
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_period)
        prophet_forecast = model.predict(future)['yhat'].tail(forecast_period).values

        # Model Selection Based on Lowest Error
        errors = {
            "ARIMA": np.mean((arima_forecast - y[-forecast_period:])**2),
            "Prophet": np.mean((prophet_forecast - y[-forecast_period:])**2)
        }
        best_model = min(errors, key=errors.get)
        best_forecast = arima_forecast if best_model == "ARIMA" else prophet_forecast
        return best_model, best_forecast

    def display_forecast(data, y, forecast):
        plt.figure(figsize=(14, 7))
        plt.plot(data['Target'], label='Historical Data', color='black')
        plt.plot(range(len(y), len(y) + forecast_period), forecast, label='Best Forecast', linestyle='--')
        plt.legend()
        st.pyplot(plt)

    def optimize_and_display_strategy(data, strategy_type):
        if strategy_type == "RSI":
            data['RSI'] = 100 - (100 / (1 + data['Target'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / abs(data['Target'].diff()).rolling(14).mean()))
            data['Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
        elif strategy_type == "MACD":
            data['MACD'] = data['Target'].ewm(span=12).mean() - data['Target'].ewm(span=26).mean()
            data['Signal'] = np.where(data['MACD'] > 0, 1, -1)
        elif strategy_type == "SMA":
            data['SMA_50'] = data['Target'].rolling(window=50).mean()
            data['SMA_200'] = data['Target'].rolling(window=200).mean()
            data['Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, -1)
        
        data['Strategy_Return'] = data['Signal'].shift(1) * data['Target'].pct_change()
        st.write(f"✅ {strategy_type} Strategy Return: {data['Strategy_Return'].cumsum().iloc[-1]:.4f}")
        st.line_chart(data['Strategy_Return'].cumsum())

    agentic_model_and_strategy(prompt, data)
else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
