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

st.set_page_config(page_title="🚀 Advanced Agentic Model Creation Tool", layout="wide")
st.title("🚀 Advanced Agentic Model Creation Tool (Enhanced + Full Feature Set)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)
auto_reoptimize = st.sidebar.checkbox("🔄 Auto Re-Optimize Models Every Hour", value=True)
refresh_interval = st.sidebar.number_input("Auto-Refresh Interval (Seconds)", min_value=60, value=3600)
custom_strategy = st.sidebar.checkbox("📝 Enable Custom Strategy Creation", value=True)

# Enhanced Smart Data Sourcing (Prompt-Based)
@st.cache_data(ttl=60 * 60)
def smart_data_sourcing(prompt):
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
    
    if ticker:
        st.write(f"✅ Fetching Real-Time Data for {ticker} (Yahoo Finance)")
        try:
            data = yf.download(ticker, period="2y", interval="1d")
            if data.empty:
                st.error(f"❌ No data found for {ticker}. Please enter a valid ticker.")
                return None
            
            data.reset_index(inplace=True)
            
            # Ensure column names are strings (handle MultiIndex)
            data.columns = [str(col) if isinstance(col, tuple) else col for col in data.columns]

            # Automatically Detect Date Column
            possible_date_columns = [col for col in data.columns if any(date_kw in str(col).lower() for date_kw in ['date', 'timestamp', 'time', 'datetime'])]
            
            if not possible_date_columns:
                st.error("❌ No 'Date' column found in the data. Unable to proceed.")
                return None

            # Use the first matching column as the Date column
            date_column = possible_date_columns[0]
            data.rename(columns={date_column: "Date"}, inplace=True)
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

            # Drop rows with invalid dates
            data.dropna(subset=['Date'], inplace=True)
            data = data[data['Date'] <= datetime.now()]  # Only keep historical dates

            return data

        except Exception as e:
            st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
            return None

    st.error("❌ Unable to detect appropriate data source. Please enter a prompt or upload a file.")
    return None

# Load Data Based on User Prompt
data = smart_data_sourcing(prompt)

if data is not None and not data.empty:
    st.write("✅ Real-Time Data Loaded")
    st.write(data.head())

    # Add Technical Indicators (Fully Vectorized RSI)
    def add_technical_indicators(df):
        # RSI Calculation (Vectorized)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD Calculation
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        return df

    data = add_technical_indicators(data)

    # Display Corrected Data (Historical Dates Only)
    st.subheader("📊 Historical Data Visualization")
    st.line_chart(data.set_index("Date")['Close'])

    # Auto-Optimization & Backtesting with Ensemble Model
    def auto_optimize(y):
        st.subheader("🔧 Auto-Optimize & Ensemble Model")

        def train_arima(y):
            model = ARIMA(y, order=(5, 1, 0))
            model_fit = model.fit()
            return model_fit.forecast(steps=forecast_period)

        def train_lstm(y):
            scaler = MinMaxScaler()
            scaled_y = scaler.fit_transform(y.reshape(-1, 1))
            model = Sequential()
            model.add(LSTM(50, return_sequences=True))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(scaled_y[:-forecast_period], scaled_y[1:1-forecast_period], epochs=5, verbose=0)
            return model.predict(scaled_y[-forecast_period:].reshape(-1, 1, 1)).flatten()

        def train_prophet(df):
            prophet_df = pd.DataFrame({'ds': df['Date'], 'y': df['Close']})
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=forecast_period)
            forecast = model.predict(future)
            return forecast['yhat'].tail(forecast_period).values

        y = data['Close'].values
        arima_preds = train_arima(y)
        lstm_preds = train_lstm(y)
        prophet_preds = train_prophet(data)

        # Ensemble (Adaptive Weighted Average)
        mse_values = [np.mean((arima_preds - y[-forecast_period:])**2), 
                      np.mean((lstm_preds - y[-forecast_period:])**2), 
                      np.mean((prophet_preds - y[-forecast_period:])**2)]
        weights = 1 / np.array(mse_values)
        weights /= weights.sum()
        ensemble_preds = (weights[0] * arima_preds + weights[1] * lstm_preds + weights[2] * prophet_preds)

        # Visualization
        st.subheader("📈 Forecasting & Ensemble Visualization")
        plt.figure(figsize=(14, 7))
        plt.plot(data['Close'], label='True Values', color='black')
        plt.plot(arima_preds, label='ARIMA', linestyle='--')
        plt.plot(lstm_preds, label='LSTM', linestyle='--')
        plt.plot(prophet_preds, label='Prophet', linestyle='--')
        plt.plot(ensemble_preds, label='Ensemble (Adaptive)', color='red', linewidth=2)
        plt.legend()
        st.pyplot(plt)

        return ensemble_preds

    ensemble_preds = auto_optimize(data['Close'].values)

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
