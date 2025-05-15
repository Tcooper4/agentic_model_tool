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
from fpdf import FPDF
import requests
import openai

st.set_page_config(page_title="🚀 Agentic Model Creation Tool (Fully Automated)", layout="wide")
st.title("🚀 Agentic Model Creation Tool (Fully Automated with GPT-4)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500', 'Classify Emails')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)

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
        ticker = "SPY"
        st.write(f"✅ Automatically Fetching Stock Data for {ticker} (Yahoo Finance)")
        try:
            data = yf.download(ticker, period="1y")
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"❌ Error fetching stock data: {str(e)}")
    
    elif "crypto" in prompt or "bitcoin" in prompt or "ethereum" in prompt:
        crypto = "bitcoin"
        st.write(f"✅ Automatically Fetching Cryptocurrency Data (CoinGecko)")
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days=365"
            response = requests.get(url).json()
            prices = response.get('prices', [])
            data = pd.DataFrame(prices, columns=["timestamp", "price"])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            return data.rename(columns={"timestamp": "Date", "price": "Close"})
        except Exception as e:
            st.error(f"❌ Error fetching crypto data: {str(e)}")

    st.error("❌ Unable to detect appropriate data source. Please enter a valid request.")
    return None

# Load Data Based on User Prompt
data = smart_data_sourcing(prompt)

if data is not None and not data.empty:
    st.write("✅ Data Loaded Automatically")
    st.write(data.head())

    # Auto-Detecting Date and Target Columns
    date_column = st.selectbox("Select Date Column", data.columns, index=0)
    target_column = st.selectbox("Select Target Column", data.columns, index=-1)

    if date_column and target_column:
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data.dropna(subset=[date_column, target_column], inplace=True)
        X = data[[date_column]]
        y = data[target_column]

        # Model Explanations (Interactive)
        st.subheader("📊 Model Explanations")
        st.write("""
        - **ARIMA (Auto Regressive Integrated Moving Average):** Best for time-series data with trends.
        - **LSTM (Long Short-Term Memory):** A type of recurrent neural network (RNN) suitable for sequential data.
        - **Prophet:** Developed by Facebook, great for data with clear seasonality.
        - **GPT-4:** Text-based model using OpenAI's GPT-4, can understand complex instructions.
        """)

        # Auto-Optimization with Model Selection
        def auto_optimize(X, y):
            st.subheader("🔧 Auto-Optimize Time-Series Model")
            model_type = st.selectbox("Select Model Type", ["Automatic", "ARIMA", "LSTM", "Prophet", "GPT-4", "Compare All"])
            best_model = None
            best_mse = float('inf')

            def train_arima(y):
                model = ARIMA(y, order=(5, 1, 0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_period)
                return model_fit, forecast

            def train_lstm(y):
                scaler = MinMaxScaler()
                scaled_y = scaler.fit_transform(y.values.reshape(-1, 1))
                X_train, y_train = [], []

                for i in range(60, len(scaled_y)):
                    X_train.append(scaled_y[i-60:i, 0])
                    y_train.append(scaled_y[i, 0])
                
                X_train, y_train = np.array(X_train), np.array(y_train)
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
                
                predictions = model.predict(X_train)
                return model, predictions

            def train_gpt(prompt):
                if openai_api_key:
                    response = openai.Completion.create(
                        engine="gpt-4",
                        prompt=prompt,
                        max_tokens=100,
                        n=1,
                        stop=None
                    )
                    return response.choices[0].text.strip()
                else:
                    st.error("❌ GPT-4 API Key Required for this Option.")
                    return None

            if model_type == "GPT-4":
                forecast = train_gpt(prompt)
                st.write(f"🔍 GPT-4 Forecast: {forecast}")
                return forecast

            if model_type in ["Compare All", "ARIMA"]:
                model, forecast = train_arima(y)
                mse = mean_squared_error(y[-forecast_period:], forecast)
                st.write(f"🔍 ARIMA MSE: {mse}")

            if model_type in ["Compare All", "LSTM"]:
                model, predictions = train_lstm(y)
                mse = mean_squared_error(y[-len(predictions):], predictions)
                st.write(f"🔍 LSTM MSE: {mse}")

            if model_type in ["Compare All", "Prophet"]:
                prophet_model = Prophet()
                prophet_df = pd.DataFrame({'ds': X[date_column], 'y': y})
                prophet_model.fit(prophet_df)
                future = prophet_model.make_future_dataframe(periods=forecast_period)
                forecast = prophet_model.predict(future)
                mse = mean_squared_error(y[-forecast_period:], forecast['yhat'][-forecast_period:])
                st.write(f"🔍 Prophet MSE: {mse}")

            return model

        best_model = auto_optimize(X, y)
        if best_model:
            st.write("✅ Best Model Optimized Automatically")

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
