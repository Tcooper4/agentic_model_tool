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

st.set_page_config(page_title="🚀 Truly Agentic Model Creation Tool", layout="wide")
st.title("🚀 Truly Agentic Model Creation Tool (Fully Adaptive)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500', 'Optimize RSI Strategy')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)
custom_strategy = st.sidebar.checkbox("📝 Enable Custom Strategy Creation", value=True)

# Agentic Data Sourcing and Column Detection
@st.cache_data(ttl=60 * 60)
def agentic_data_sourcing(prompt):
    prompt = prompt.lower()
    ticker = None
    
    # Automatically detect ticker
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
    try:
        data = yf.download(ticker, period="2y", interval="1d")
        if data.empty:
            st.error(f"❌ No data found for {ticker}. Please enter a valid ticker.")
            return None
        
        data.reset_index(inplace=True)
        data.columns = [str(col) for col in data.columns]

        # Detecting the most relevant column for analysis
        possible_columns = ['Close', 'Adj Close', 'Last', 'Price']
        column = next((col for col in data.columns if any(key.lower() in col.lower() for key in possible_columns)), None)

        if not column:
            st.error("❌ Unable to identify a suitable column for analysis (e.g., Close, Price).")
            return None
        
        st.write(f"✅ Using '{column}' column for analysis.")
        data.rename(columns={column: "Target"}, inplace=True)
        return data

    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None

# Load Data Based on User Prompt
data = agentic_data_sourcing(prompt)

if data is not None and not data.empty:
    st.write("✅ Real-Time Data Loaded")
    st.write(data.head())

    # Agentic Model and Strategy Selection
    def agentic_model_selection(prompt, data):
        if "forecast" in prompt:
            st.subheader("📈 Forecasting with Advanced Models")
            y = data['Target'].values
            
            arima_model = ARIMA(y, order=(5, 1, 0)).fit()
            arima_forecast = arima_model.forecast(steps=forecast_period)
            
            model = Prophet()
            prophet_df = pd.DataFrame({'ds': data['Date'], 'y': data['Target']})
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=forecast_period)
            prophet_forecast = model.predict(future)['yhat'].tail(forecast_period).values
            
            st.write("✅ Forecasting Completed")
            plt.figure(figsize=(14, 7))
            plt.plot(data['Target'], label='Historical Data', color='black')
            plt.plot(range(len(y), len(y) + forecast_period), arima_forecast, label='ARIMA', linestyle='--')
            plt.plot(range(len(y), len(y) + forecast_period), prophet_forecast, label='Prophet', linestyle='--')
            plt.legend()
            st.pyplot(plt)

        if "rsi" in prompt:
            st.subheader("📊 RSI Strategy Optimization")
            data['RSI'] = 100 - (100 / (1 + data['Target'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / abs(data['Target'].diff()).rolling(14).mean()))
            data['RSI_Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
            data['RSI_Strategy_Return'] = data['RSI_Signal'].shift(1) * data['Target'].pct_change()
            st.write(f"✅ RSI Strategy Return: {data['RSI_Strategy_Return'].cumsum().iloc[-1]:.4f}")

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

    agentic_model_selection(prompt, data)
else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
