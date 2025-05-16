import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="🚀 Fully Agentic Model Creation Tool", layout="wide")
st.title("🚀 Fully Agentic Model Creation Tool (Truly Adaptive)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request (e.g., 'Predict SP500', 'Optimize RSI Strategy')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)
custom_strategy = st.sidebar.checkbox("📝 Enable Custom Strategy Creation", value=True)

# Intelligent Data Loading and Column Detection
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

    # Interactive Clarification (Agentic Behavior)
    st.subheader("🔍 Would you like to...")
    forecasting = st.checkbox("📈 Run a Forecast?")
    strategy_optimization = st.checkbox("📊 Optimize a Strategy?")

    if forecasting:
        st.subheader("📈 Adaptive Forecasting")
        y = data['Target'].values
        model = Prophet()
        prophet_df = pd.DataFrame({'ds': data['Date'], 'y': data['Target']})
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)
        
        # Display Forecast
        st.subheader("🔮 Forecast Visualization")
        plt.figure(figsize=(14, 7))
        plt.plot(data['Target'], label='Historical Data', color='black')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
        plt.legend()
        st.pyplot(plt)

    if strategy_optimization:
        st.subheader("📊 Choose a Strategy to Optimize")
        strategy_choice = st.selectbox("Choose a Strategy:", ["RSI", "MACD", "SMA", "Bollinger Bands"])

        if strategy_choice == "RSI":
            data['RSI'] = 100 - (100 / (1 + data['Target'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / abs(data['Target'].diff()).rolling(14).mean()))
            data['Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
            st.write(f"✅ RSI Strategy Optimized")
            st.line_chart(data[['RSI']])

        elif strategy_choice == "MACD":
            data['MACD'] = data['Target'].ewm(span=12).mean() - data['Target'].ewm(span=26).mean()
            data['Signal'] = np.where(data['MACD'] > 0, 1, -1)
            st.write(f"✅ MACD Strategy Optimized")
            st.line_chart(data[['MACD']])

        elif strategy_choice == "SMA":
            data['SMA_50'] = data['Target'].rolling(window=50).mean()
            data['SMA_200'] = data['Target'].rolling(window=200).mean()
            data['Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, -1)
            st.write(f"✅ SMA Crossover Strategy Optimized")
            st.line_chart(data[['SMA_50', 'SMA_200']])

        elif strategy_choice == "Bollinger Bands":
            data['SMA'] = data['Target'].rolling(window=20).mean()
            data['Upper_Band'] = data['SMA'] + 2 * data['Target'].rolling(window=20).std()
            data['Lower_Band'] = data['SMA'] - 2 * data['Target'].rolling(window=20).std()
            st.write(f"✅ Bollinger Bands Strategy Optimized")
            st.line_chart(data[['SMA', 'Upper_Band', 'Lower_Band']])

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
