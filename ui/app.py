import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import openai
import time

st.set_page_config(page_title="🚀 Advanced Agentic Model Creation Tool", layout="wide")
st.title("🚀 Advanced Agentic Model Creation Tool (Real-Time Data, Backtesting, Strategy Optimization, Auto-Switching)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)
auto_reoptimize = st.sidebar.checkbox("🔄 Auto Re-Optimize Models Every Hour", value=True)
refresh_interval = st.sidebar.number_input("Auto-Refresh Interval (Seconds)", min_value=60, value=3600)
custom_strategy = st.sidebar.checkbox("📝 Enable Custom Strategy Creation", value=True)

# Choose LLM (Hugging Face or OpenAI)
llm_type = st.sidebar.selectbox("🔑 Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])
if llm_type == "OpenAI (GPT-4)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Required for GPT-4)", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key
        st.sidebar.write("💡 Estimated Cost: ~$0.03 per 1,000 tokens")

# Enhanced Smart Data Sourcing (Real-Time)
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
            return data
        except Exception as e:
            st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
            return None

    st.error("❌ Unable to detect appropriate data source. Please enter a valid request.")
    return None

# Load Data Based on User Prompt
data = smart_data_sourcing(prompt)

if data is not None and not data.empty:
    st.write("✅ Real-Time Data Loaded")
    st.write(data.head())

    # Target Column Selection
    target_column = st.selectbox("Select Target Column", data.columns, index=data.columns.get_loc("Close") if "Close" in data.columns else -1)

    if target_column:
        y = data[target_column].values
        data['Date'] = pd.to_datetime(data['Date'])
        X = data[['Date']]

        # Add Technical Indicators
        def add_technical_indicators(df):
            df['RSI'] = df['Close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / abs(df['Close'].diff()).rolling(14).mean() * 100
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            return df

        data = add_technical_indicators(data)

        # Auto-Optimization & Backtesting with Ensemble Model
        def auto_optimize(X, y):
            st.subheader("🔧 Auto-Optimize & Ensemble Model")
            models = []

            def train_arima(y):
                model = ARIMA(y, order=(5, 1, 0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_period)
                return forecast

            def train_lstm(y):
                scaler = MinMaxScaler()
                scaled_y = scaler.fit_transform(y.reshape(-1, 1))
                X_train, y_train = [], []

                for i in range(60, len(scaled_y)):
                    X_train.append(scaled_y[i-60:i, 0])
                    y_train.append(scaled_y[i, 0])
                
                X_train = np.array(X_train)
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                y_train = np.array(y_train)

                model = Sequential()
                model.add(LSTM(50, return_sequences=True))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
                preds = model.predict(X_train)
                return preds.flatten()

            def train_prophet(X, y):
                prophet_df = pd.DataFrame({'ds': X['Date'], 'y': y})
                model = Prophet()
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=forecast_period)
                forecast = model.predict(future)
                return forecast['yhat'][-forecast_period:]

            arima_preds = train_arima(y)
            lstm_preds = train_lstm(y)
            prophet_preds = train_prophet(X, y)

            # Ensemble Forecast (Weighted Average)
            ensemble_preds = (arima_preds + lstm_preds[:len(arima_preds)] + prophet_preds) / 3

            # Strategy Optimization (SMA Crossover)
            data['Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)
            data['Strategy_Return'] = data['Signal'].shift(1) * data['Close'].pct_change()
            strategy_performance = data['Strategy_Return'].cumsum()

            # Visualization
            st.subheader("📈 Forecasting & Strategy Visualization")
            plt.figure(figsize=(14, 7))
            plt.plot(data['Close'], label='True Values', color='black')
            plt.plot(arima_preds, label='ARIMA', linestyle='--')
            plt.plot(lstm_preds[:len(arima_preds)], label='LSTM', linestyle='--')
            plt.plot(prophet_preds, label='Prophet', linestyle='--')
            plt.plot(ensemble_preds, label='Ensemble', color='red', linewidth=2)
            plt.legend()
            st.pyplot(plt)

            st.subheader("🔍 Strategy Backtesting Results")
            st.write(f"✅ Strategy Performance: {strategy_performance.iloc[-1]:.4f}")

            # Custom Strategy (Optional)
            if custom_strategy:
                st.subheader("📝 Custom Strategy Creation")
                custom_signal = st.text_input("Enter Custom Strategy Logic (e.g., 'RSI > 70 and MACD > 0')")
                if custom_signal:
                    data['Custom_Signal'] = data.eval(custom_signal).astype(int)
                    data['Custom_Strategy_Return'] = data['Custom_Signal'].shift(1) * data['Close'].pct_change()
                    st.write(f"✅ Custom Strategy Performance: {data['Custom_Strategy_Return'].cumsum().iloc[-1]:.4f}")

        auto_optimize(X, y)
else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
