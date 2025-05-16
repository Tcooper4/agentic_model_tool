import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="🚀 Advanced Agentic Model Creation Tool", layout="wide")
st.title("🚀 Advanced Agentic Model Creation Tool (Real-Time Data, Auto-Run, Best Strategy Detection)")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
prompt = st.sidebar.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500')", "")
forecast_period = st.sidebar.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=7)
auto_run = st.sidebar.checkbox("🔄 Auto-Run Strategies Daily", value=True)
save_file = "optimized_strategies.json"
log_file = "strategy_performance_log.csv"

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

    # Load Saved Strategies
    def load_optimized_strategies():
        if os.path.exists(save_file):
            with open(save_file, "r") as file:
                return json.load(file)
        return {}

    optimized_strategies = load_optimized_strategies()
    st.subheader("✅ Loaded Optimized Strategies")
    st.write(optimized_strategies)

    # Apply and Evaluate Strategies
    def apply_strategy(df, strategy, params):
        if strategy == "RSI Overbought/Oversold":
            period = params
            df['RSI'] = df['Close'].diff().apply(lambda x: max(x, 0)).rolling(period).mean() / abs(df['Close'].diff()).rolling(period).mean() * 100
            df['Signal'] = np.where(df['RSI'] > 70, -1, np.where(df['RSI'] < 30, 1, 0))
        elif strategy == "MACD Crossover":
            fast, slow = params
            df['MACD'] = df['Close'].ewm(span=fast).mean() - df['Close'].ewm(span=slow).mean()
            df['Signal'] = np.where(df['MACD'] > 0, 1, -1)

        df['Strategy_Return'] = df['Signal'].shift(1) * df['Close'].pct_change()
        df['Cumulative_Return'] = df['Strategy_Return'].cumsum()
        return df

    # Strategy Performance Logging
    def log_strategy_performance(strategy, params, performance):
        if not os.path.exists(log_file):
            with open(log_file, "w") as file:
                file.write("Date,Strategy,Parameters,Cumulative_Return\n")
        
        with open(log_file, "a") as file:
            file.write(f"{datetime.now().strftime('%Y-%m-%d')},{strategy},{json.dumps(params)},{performance:.4f}\n")

    # Auto-Run Strategies Daily
    if auto_run:
        for strategy, params in optimized_strategies.items():
            data = apply_strategy(data, strategy, params)
            cumulative_return = data['Cumulative_Return'].iloc[-1]
            log_strategy_performance(strategy, params, cumulative_return)
        st.success("✅ Auto-Run Completed: Strategies have been logged.")

    # Display Strategy Comparison
    st.subheader("📊 Strategy Comparison")
    fig = go.Figure()

    for strategy, params in optimized_strategies.items():
        data = apply_strategy(data, strategy, params)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Cumulative_Return'], name=f"{strategy}"))

    st.plotly_chart(fig, use_container_width=True)

    # Best Strategy of All-Time
    st.subheader("🏆 Best Strategy of All-Time")
    if os.path.exists(log_file):
        performance_log = pd.read_csv(log_file)
        best_strategy = performance_log.groupby("Strategy").Cumulative_Return.mean().idxmax()
        best_return = performance_log.groupby("Strategy").Cumulative_Return.mean().max()
        st.write(f"✅ Best Strategy: {best_strategy} with Average Return: {best_return:.4f}")
        st.write(performance_log)
    else:
        st.write("❌ No performance data available yet.")

    # Save and Manage Strategies
    st.subheader("💾 Save and Manage Strategies")
    if st.button("Save Optimized Strategies"):
        with open(save_file, "w") as file:
            json.dump(optimized_strategies, file)
        st.success("✅ Strategies Saved!")

    if st.button("Clear Performance Log"):
        if os.path.exists(log_file):
            os.remove(log_file)
        st.success("✅ Performance Log Cleared!")

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
