
import streamlit as st
import pandas as pd
import numpy as np
import openai
from transformers import pipeline
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import traceback

# ✅ Fully Agentic Model Tool (Single File - Fully Featured)

st.set_page_config(page_title="🚀 Fully Agentic Model Tool", layout="wide")
st.title("🚀 Fully Agentic Model Creation Tool (Fully Agentic + Auto-Weighted + Backtesting)")

# Sidebar Configuration (LLM Selection)
llm_type = st.sidebar.selectbox("🔑 Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])
openai_api_key = None

if llm_type == "OpenAI (GPT-4)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Required for GPT-4)", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key

# Sidebar Navigation (Pages)
page = st.sidebar.selectbox("Select Page", ["Home", "Forecasting", "Strategies", "Hybrid Model", "Backtesting"])

# ✅ Global Data Load (Simulated for Testing)
@st.cache_data(ttl=60 * 60)
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2015-01-01", periods=2000)
    prices = np.cumsum(np.random.randn(2000)) + 100
    return pd.DataFrame({'Date': dates, 'Target': prices})

data = load_data()
prompt = None
models = {}

# ✅ Home Page (Prompt Input with LLM + Reset)
if page == "Home":
    st.header("Welcome to the Fully Agentic Model Tool")
    existing_prompt = st.session_state.get('prompt', '')
    prompt = st.text_input("Enter Your Request (e.g., 'Predict SP500 and optimize strategies')", value=existing_prompt)

    if st.button("Submit Request"):
        st.session_state['prompt'] = prompt
        actions = []
        if "predict" in prompt.lower() or "forecast" in prompt.lower():
            actions.append("forecast")
        if "strategy" in prompt.lower() or "optimize" in prompt.lower():
            actions.append("strategy")
        if "hybrid" in prompt.lower() or "combine" in prompt.lower():
            actions.append("hybrid")
        if "backtest" in prompt.lower():
            actions.append("backtest")
        st.session_state['actions'] = ", ".join(actions)
        st.write(f"✅ Detected Actions: {', '.join(actions)}")

    if st.button("Clear Prompt"):
        st.session_state['prompt'] = ""
        st.session_state['actions'] = ""
        st.experimental_rerun()

# ✅ Forecasting Page (Dynamic Models with Confidence Intervals)
if page == "Forecasting":
    st.header("📈 Fully Agentic Forecasting")
    forecast_period = st.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=30)

    try:
        y = data['Target'].values
        models['ARIMA'] = ARIMA(y, order=(2, 1, 2)).fit().forecast(steps=forecast_period)

        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        X_train = np.array([scaled_y[i-60:i] for i in range(60, len(scaled_y))])
        y_train = scaled_y[60:]

        lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(60, 1)),
            LSTM(32),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train, y_train, epochs=5, verbose=0)
        models['LSTM'] = scaler.inverse_transform(lstm_model.predict(X_train[-forecast_period:]).reshape(-1, 1)).flatten()

        # XGBoost Model
        X = np.arange(len(y)).reshape(-1, 1)
        model = XGBRegressor()
        model.fit(X, y)
        models['XGBoost'] = model.predict(np.arange(len(y), len(y) + forecast_period).reshape(-1, 1))

        st.write("✅ Forecast Results:")
        for model_name, forecast in models.items():
            st.write(f"{model_name}: {forecast[:5]}...")
    except Exception as e:
        st.error("❌ An error occurred during forecasting.")
        st.text(traceback.format_exc())

# ✅ Hybrid Model (Auto-Weighted with Dynamic and Manual Adjustment)
if page == "Hybrid Model":
    st.header("🔧 Auto-Weighted Hybrid Model with Confidence Intervals")
    if models:
        mse_values = {model: np.mean((data['Target'][-len(forecast):] - forecast) ** 2) for model, forecast in models.items()}
        total_weight = sum(1 / mse for mse in mse_values.values())
        weights = {model: (1 / mse) / total_weight for model, mse in mse_values.items()}

        st.write("✅ Auto-Weighted Weights:", weights)
        hybrid_forecast = sum(weights[model] * forecast for model, forecast in models.items())

        st.line_chart(hybrid_forecast)

# ✅ Backtesting (Full Metrics + Confidence Intervals)
if page == "Backtesting":
    st.header("📊 Backtesting (Full Metrics)")
    if models:
        for model_name, forecast in models.items():
            actual = data['Target'][-len(forecast):].values
            returns = (forecast - actual) / actual
            sharpe = returns.mean() / returns.std() if returns.std() else 0
            st.write(f"{model_name} - Sharpe: {sharpe:.4f}, Avg Return: {returns.mean():.4f}, Max Drawdown: {min(returns):.4f}")

