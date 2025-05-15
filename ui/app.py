import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
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
import optuna
import threading
import time

st.set_page_config(page_title="🚀 Advanced Agentic Model Creation Tool", layout="wide")
st.title("🚀 Advanced Agentic Model Creation Tool (Fully Automated with Hybrid Model)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)
auto_mode = st.sidebar.checkbox("🌐 Fully Automated Mode", value=True)
auto_update = st.sidebar.checkbox("🔁 Enable Auto-Update (Every 24 hours)", value=False)

# Choose LLM (Hugging Face or OpenAI)
llm_type = st.sidebar.selectbox("🔑 Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])
if llm_type == "OpenAI (GPT-4)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Required for GPT-4)", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key
        st.sidebar.write("💡 Estimated Cost: ~$0.03 per 1,000 tokens")

# Auto Data Sourcing Based on Prompt
def smart_data_sourcing(prompt):
    prompt = prompt.lower()
    if "sp500" in prompt or "stock" in prompt or "ticker" in prompt:
        ticker = prompt.split()[-1] if " " in prompt else "SPY"
        st.write(f"✅ Automatically Fetching Stock Data for {ticker} (Yahoo Finance)")
        try:
            data = yf.download(ticker, period="2y")
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"❌ Error fetching stock data: {str(e)}")
    st.error("❌ Unable to detect appropriate data source. Please enter a valid request.")
    return None

# Load Data Based on User Prompt
data = smart_data_sourcing(prompt)

def auto_refresh_forecast():
    while True:
        if auto_update:
            st.experimental_rerun()
        time.sleep(86400)  # Auto-update every 24 hours

# Start auto-refresh thread if enabled
if auto_update:
    threading.Thread(target=auto_refresh_forecast, daemon=True).start()

if data is not None and not data.empty:
    st.write("✅ Data Loaded Automatically")
    st.write(data.head())

    # Allow user to select target column (Price)
    target_column = st.selectbox("Select Target Column", data.columns, index=data.columns.get_loc("Close") if "Close" in data.columns else -1)

    if target_column:
        y = data[target_column].values
        data['Date'] = pd.to_datetime(data['Date'])
        X = data[['Date']]

        st.subheader("📊 Model Explanations")
        st.write("""
        - **ARIMA (Auto Regressive Integrated Moving Average):** Best for time-series data with trends.
        - **LSTM (Long Short-Term Memory):** A type of recurrent neural network (RNN) suitable for sequential data.
        - **Prophet:** Developed by Facebook, great for data with clear seasonality.
        - **Hybrid Model:** Weighted combination of the best models for improved accuracy.
        """)

        # Auto-Optimization with Model Selection
        def auto_optimize(X, y):
            st.subheader("🔧 Auto-Optimize Model")
            best_models = []
            best_mse = float('inf')

            def calculate_confidence(mse):
                return max(0, 100 - mse * 100)  # Inverse relationship

            def train_arima(y):
                model = ARIMA(y, order=(5, 1, 0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_period)
                mse = mean_squared_error(y[-forecast_period:], forecast)
                confidence = calculate_confidence(mse)
                return model_fit, forecast, mse, confidence

            def train_lstm(y):
                scaler = MinMaxScaler()
                scaled_y = scaler.fit_transform(y.reshape(-1, 1))
                X_train, y_train = [], []

                for i in range(60, len(scaled_y)):
                    X_train.append(scaled_y[i-60:i, 0])
                    y_train.append(scaled_y[i, 0])
                
                X_train, y_train = np.array(X_train), np.array(y_train)
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

                model = Sequential()
                model.add(LSTM(50, return_sequences=True))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
                predictions = model.predict(X_train)
                mse = mean_squared_error(y[-len(predictions):], scaler.inverse_transform(predictions))
                confidence = calculate_confidence(mse)
                return model, predictions, mse, confidence

            def train_prophet(X, y):
                prophet_df = pd.DataFrame({'ds': X['Date'], 'y': y})
                model = Prophet()
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=forecast_period)
                forecast = model.predict(future)
                mse = mean_squared_error(y[-forecast_period:], forecast['yhat'][-forecast_period:])
                confidence = calculate_confidence(mse)
                return model, forecast['yhat'], mse, confidence

            for model_type in ["ARIMA", "LSTM", "Prophet"]:
                if model_type == "ARIMA":
                    model, forecast, mse, confidence = train_arima(y)
                elif model_type == "LSTM":
                    model, forecast, mse, confidence = train_lstm(y)
                else:
                    model, forecast, mse, confidence = train_prophet(X, y)

                st.write(f"🔍 {model_type} MSE: {mse:.4f}, Confidence: {confidence:.2f}%")
                best_models.append((model_type, forecast, mse, confidence))

            return best_models

        # Auto-Optimize and Select Best Model
        best_models = auto_optimize(X, y)
        st.subheader("✅ Best Model Comparison with Hybrid Model")
        
        hybrid_forecast = np.zeros(forecast_period)
        total_weight = 0
        for model_name, forecast, mse, confidence in best_models:
            weight = 1 / mse if mse > 0 else 1
            hybrid_forecast += np.array(forecast[:forecast_period]) * weight
            total_weight += weight
            st.write(f"🔍 {model_name}: MSE: {mse:.4f}, Confidence: {confidence:.2f}%")

        hybrid_forecast /= total_weight
        st.write("### 🚀 Hybrid Model Forecast (Weighted Average)")
        st.line_chart(hybrid_forecast)

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
