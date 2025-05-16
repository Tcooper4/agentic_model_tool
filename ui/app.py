
import streamlit as st
import pandas as pd
import numpy as np
import openai
from transformers import pipeline
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from prophet import Prophet
from arch import arch_model  # GARCH Model (Auto-Detected)
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import traceback
import requests
import importlib

st.set_page_config(page_title="🚀 Fully Interactive Agentic Model Tool", layout="wide")
st.title("🚀 Fully Interactive Agentic Model Tool (Dynamic Hybrid + Adjustable Weights)")

# Sidebar Configuration (LLM Selection)
llm_type = st.sidebar.selectbox("🔑 Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])

# Sidebar Navigation (Pages)
page = st.sidebar.selectbox("Select Page", ["Home", "Forecasting", "Hybrid Model", "Backtesting", "Model Discovery Log"])

@st.cache_data(ttl=60 * 60)
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2015-01-01", periods=2000)
    prices = np.cumsum(np.random.randn(2000)) + 100
    return pd.DataFrame({'Date': dates, 'Target': prices})

data = load_data()
models = {}
model_log = []

# ✅ Agentic System (Self-Improving + Full Forecast + Metrics)
def model_builder_agent(y):
    models = {}
    log = []

    try:
        models['ARIMA'] = ARIMA(y, order=(2, 1, 2)).fit().forecast(steps=30)
        log.append("ARIMA: Successfully Built")
    except Exception as e:
        log.append(f"ARIMA Error: {str(e)}")

    try:
        scaler = MinMaxScaler()
        scaled_y = scaler.fit_transform(y.reshape(-1, 1))
        X_train = np.array([scaled_y[i-60:i] for i in range(60, len(scaled_y))])
        y_train = scaled_y[60:]
        lstm_model = Sequential([LSTM(64, return_sequences=True, input_shape=(60, 1)), LSTM(32), Dense(1)])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train, y_train, epochs=5, verbose=0)
        lstm_forecast = scaler.inverse_transform(lstm_model.predict(X_train[-30:]).reshape(-1, 1)).flatten()
        models['LSTM'] = lstm_forecast
        log.append("LSTM: Successfully Built")
    except Exception as e:
        log.append(f"LSTM Error: {str(e)}")

    try:
        model = XGBRegressor()
        model.fit(np.arange(len(y)).reshape(-1, 1), y)
        models['XGBoost'] = model.predict(np.arange(len(y), len(y) + 30).reshape(-1, 1))
        log.append("XGBoost: Successfully Built")
    except Exception as e:
        log.append(f"XGBoost Error: {str(e)}")

    model_log.extend(log)
    return models

# ✅ Home Page (Prompt Input)
if page == "Home":
    st.header("Welcome to the Fully Interactive Agentic Model Tool")
    prompt = st.text_input("Enter Your Request (e.g., 'Predict SP500 and optimize strategies')")
    if st.button("Submit Request"):
        st.session_state['prompt'] = prompt

# ✅ Forecasting Page (Dynamic + Full Forecast Display)
if page == "Forecasting":
    st.header("📈 Fully Autonomous Forecasting")
    y = data['Target'].values
    models = model_builder_agent(y)
    st.write("✅ Models Built:", list(models.keys()))

    forecast_df = pd.DataFrame({model: forecast for model, forecast in models.items()})
    st.subheader("🔍 Forecast Values")
    st.write(forecast_df.head(10))

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Target'], mode='lines', name='Actual'))
    for model_name, forecast in models.items():
        fig.add_trace(go.Scatter(x=data['Date'].tail(len(forecast)), y=forecast, mode='lines', name=model_name))
    fig.update_layout(title="Model Forecasts (Interactive)", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

# ✅ Hybrid Model (Auto-Weighted + Adjustable)
if page == "Hybrid Model":
    st.header("🔧 Auto-Weighted Hybrid Model (Adjustable)")
    if models:
        st.write("✅ Models Detected:", list(models.keys()))

        # Adjustable Weights
        st.subheader("🔧 Adjust Model Weights")
        weights = {model: st.slider(f"{model} Weight", 0.0, 1.0, 1 / len(models)) for model in models}

        # Normalizing Weights
        total_weight = sum(weights.values())
        weights = {model: weight / total_weight for model, weight in weights.items()}
        st.write("✅ Model Weights (Normalized):", weights)

        # Hybrid Forecast
        hybrid_forecast = sum(weights[model] * forecast for model, forecast in models.items())
        st.subheader("🔍 Hybrid Forecast Values")
        st.write(pd.Series(hybrid_forecast).head(10))

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Target'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=data['Date'].tail(len(hybrid_forecast)), y=hybrid_forecast, mode='lines', name='Hybrid Model'))
        st.plotly_chart(fig)

# ✅ Model Discovery Log Page
if page == "Model Discovery Log":
    st.header("📊 Model Discovery Log")
    st.write(model_log)
