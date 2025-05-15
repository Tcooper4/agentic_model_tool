import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import openai

st.set_page_config(page_title="🚀 Agentic Model Creation Tool (Fully Automated)", layout="wide")
st.title("🚀 Agentic Model Creation Tool (Fully Automated with GPT-4)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500', 'Predict AAPL', 'Classify Emails')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)

# Choose LLM (Hugging Face or OpenAI)
llm_type = st.sidebar.selectbox("🔑 Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])
if llm_type == "OpenAI (GPT-4)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Required for GPT-4)", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key
        st.sidebar.write("💡 Estimated Cost: ~$0.03 per 1,000 tokens")

# Enhanced Smart Data Sourcing with Error Handling
def smart_data_sourcing(prompt):
    prompt = prompt.lower()
    
    # Detect Stock Ticker
    if "predict" in prompt:
        words = prompt.split()
        for word in words:
            if len(word) <= 5 and word.isalpha():
                ticker = word.upper()
                st.write(f"✅ Automatically Fetching Stock Data for {ticker} (Yahoo Finance)")
                try:
                    data = yf.download(ticker, period="1y")
                    data.reset_index(inplace=True)
                    if not data.empty:
                        return data
                except Exception as e:
                    st.error(f"❌ Error fetching stock data for {ticker}: {str(e)}")

    # Cryptocurrency Data
    if "crypto" in prompt or "bitcoin" in prompt or "ethereum" in prompt:
        crypto = "bitcoin" if "bitcoin" in prompt else "ethereum"
        st.write(f"✅ Automatically Fetching {crypto.capitalize()} Data (CoinGecko)")
        url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days=365"
        response = requests.get(url).json()
        prices = response.get('prices', [])
        if prices:
            data = pd.DataFrame(prices, columns=["timestamp", "price"])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            return data.rename(columns={"timestamp": "Date", "price": "Close"})

    st.error("❌ Unable to detect appropriate data source. Please enter a valid request.")
    return None

# Load Data Based on User Prompt
data = smart_data_sourcing(prompt)

if data is not None and not data.empty:
    st.write("✅ Data Loaded Automatically")
    st.write(data.head())

    # Ensure the DataFrame has columns
    if len(data.columns) == 0:
        st.error("❌ Data has no columns. Please try a different source.")
    else:
        # Auto-Detecting Date and Target Columns Safely
        date_column = st.selectbox("Select Date Column", data.columns, index=0)
        target_column = st.selectbox("Select Target Column", data.columns, index=len(data.columns) - 1)

        if date_column and target_column:
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
            data.dropna(subset=[date_column, target_column], inplace=True)
            X = data[[date_column]]
            y = data[target_column]

            # Model Comparison Function
            def compare_models(X, y):
                st.subheader("📊 Comparing Models")
                results = {}

                # ARIMA
                try:
                    model = ARIMA(y, order=(5, 1, 0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=forecast_period)
                    mse = mean_squared_error(y[-forecast_period:], forecast)
                    results['ARIMA'] = mse
                    st.write(f"🔍 ARIMA MSE: {mse}")
                except Exception as e:
                    st.write(f"❌ ARIMA Error: {str(e)}")

                # LSTM
                try:
                    scaler = MinMaxScaler()
                    scaled_y = scaler.fit_transform(np.array(y).reshape(-1, 1))
                    X_train, y_train = [], []

                    for i in range(60, len(scaled_y)):
                        X_train.append(scaled_y[i-60:i, 0])
                        y_train.append(scaled_y[i, 0])

                    X_train, y_train = np.array(X_train), np.array(y_train)
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

                    lstm_model = Sequential()
                    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                    lstm_model.add(LSTM(50))
                    lstm_model.add(Dense(1))
                    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
                    lstm_model.fit(X_train, y_train, epochs=5, verbose=0)

                    predictions = lstm_model.predict(X_train)
                    mse = mean_squared_error(y_train, predictions)
                    results['LSTM'] = mse
                    st.write(f"🔍 LSTM MSE: {mse}")
                except Exception as e:
                    st.write(f"❌ LSTM Error: {str(e)}")

                # Prophet
                try:
                    prophet_df = pd.DataFrame({'ds': data[date_column], 'y': y})
                    prophet_model = Prophet()
                    prophet_model.fit(prophet_df)
                    future = prophet_model.make_future_dataframe(periods=forecast_period)
                    forecast = prophet_model.predict(future)
                    mse = mean_squared_error(y[-forecast_period:], forecast['yhat'][-forecast_period:])
                    results['Prophet'] = mse
                    st.write(f"🔍 Prophet MSE: {mse}")
                except Exception as e:
                    st.write(f"❌ Prophet Error: {str(e)}")

                return results

            # Automatically Compare Models
            model_results = compare_models(X, y)
            if model_results:
                best_model = min(model_results, key=model_results.get)
                st.write(f"✅ Best Model: **{best_model}** with MSE: {model_results[best_model]}")
            else:
                st.error("❌ No models successfully trained.")

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
