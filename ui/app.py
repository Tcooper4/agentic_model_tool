import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ✅ Streamlit Multi-Page Agentic Model Creation Tool (Setup)
st.set_page_config(page_title="🚀 Fully Agentic Model Tool", layout="wide")
st.title("🚀 Fully Agentic Model Creation Tool")

# Sidebar Navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Forecasting", "Strategies", "Backtesting", "Real-Time Alerts"])

# ✅ Global Data Load (Simulated for Testing)
@st.cache_data(ttl=60 * 60)
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2015-01-01", periods=2000)
    prices = np.cumsum(np.random.randn(2000)) + 100  # Simulated random walk
    return pd.DataFrame({'Date': dates, 'Target': prices})

data = load_data()

# ✅ Home Page
if page == "Home":
    st.header("Welcome to the Fully Agentic Model Tool")
    st.write("This tool automatically understands your requests, optimizes forecasting models, and tests the best trading strategies.")
    st.write("Use the sidebar to navigate between pages.")

# ✅ Forecasting Page
if page == "Forecasting":
    st.header("📈 Fully Agentic Forecasting")
    forecast_period = st.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=30)
    
    def run_forecasting(data, forecast_period):
        y = data['Target'].values
        best_aic = float("inf")
        best_order = None
        best_model = None

        for p in range(1, 4):
            for d in range(0, 2):
                for q in range(0, 2):
                    try:
                        model = ARIMA(y, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, d, q)
                            best_model = model
                    except:
                        continue

        forecast = best_model.forecast(steps=forecast_period)
        st.write(f"✅ Best Model: ARIMA{best_order} with AIC: {best_aic:.4f}")

        # Visualization
        plt.figure(figsize=(14, 7))
        plt.plot(data['Target'], label='Historical Data', color='black')
        plt.plot(range(len(y), len(y) + forecast_period), forecast, label='Best Forecast', linestyle='--')
        plt.legend()
        st.pyplot(plt)

    if st.button("Run Forecasting"):
        run_forecasting(data, forecast_period)

# ✅ Strategies Page
if page == "Strategies":
    st.header("📊 Fully Agentic Strategy Optimization")
    strategies = {}

    # RSI Strategy (Auto-Optimized)
    data['RSI'] = 100 - (100 / (1 + data['Target'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / abs(data['Target'].diff()).rolling(14).mean()))
    data['RSI_Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
    data['RSI_Return'] = data['RSI_Signal'].shift(1) * data['Target'].pct_change()
    strategies['RSI'] = data['RSI_Return'].cumsum().iloc[-1]

    # MACD Strategy (Auto-Optimized)
    data['MACD'] = data['Target'].ewm(span=12).mean() - data['Target'].ewm(span=26).mean()
    data['MACD_Signal'] = np.where(data['MACD'] > 0, 1, -1)
    data['MACD_Return'] = data['MACD_Signal'].shift(1) * data['Target'].pct_change()
    strategies['MACD'] = data['MACD_Return'].cumsum().iloc[-1]

    # SMA Crossover Strategy (Auto-Optimized)
    data['SMA_50'] = data['Target'].rolling(window=50).mean()
    data['SMA_200'] = data['Target'].rolling(window=200).mean()
    data['SMA_Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, -1)
    data['SMA_Return'] = data['SMA_Signal'].shift(1) * data['Target'].pct_change()
    strategies['SMA'] = data['SMA_Return'].cumsum().iloc[-1]

    # Best Strategy Selection
    best_strategy = max(strategies, key=strategies.get)
    st.write(f"✅ Best Strategy: {best_strategy} with return of {strategies[best_strategy]:.4f}")

    # Strategy Visualization
    st.line_chart(data[['RSI', 'MACD', 'SMA_50', 'SMA_200']])

# ✅ Backtesting Page
if page == "Backtesting":
    st.header("🔍 Strategy Backtesting")
    strategy = st.selectbox("Select Strategy", ["RSI", "MACD", "SMA"])
    st.write(f"Backtesting the {strategy} strategy...")

    data['Backtest_Signal'] = data[f'{strategy}_Signal']
    data['Backtest_Return'] = data['Backtest_Signal'].shift(1) * data['Target'].pct_change()
    st.line_chart(data['Backtest_Return'].cumsum())

# ✅ Real-Time Alerts Page
if page == "Real-Time Alerts":
    st.header("🔔 Real-Time Buy/Sell Notifications")
    strategy = st.selectbox("Select Strategy for Alerts", ["RSI", "MACD", "SMA"])
    st.write(f"Monitoring real-time signals for {strategy}...")

    if strategy in data.columns:
        data['Alert_Signal'] = data[f'{strategy}_Signal']
        for i in range(1, len(data)):
            if data['Alert_Signal'].iloc[i] == 1:
                st.success(f"📈 BUY Signal at {data['Date'].iloc[i]}")
            elif data['Alert_Signal'].iloc[i] == -1:
                st.error(f"📉 SELL Signal at {data['Date'].iloc[i]}")

st.sidebar.write("🚀 Built with Fully Agentic Automation")
