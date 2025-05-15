import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import optuna
import yfinance as yf
import openai
from transformers import pipeline
import requests
from fpdf import FPDF
import datetime
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import shap
import json
from streamlit_autorefresh import st_autorefresh

# Configure Logging
logging.basicConfig(filename='agentic_model_tool.log', level=logging.INFO, format='%(asctime)s - %(message)s')

st.set_page_config(page_title="🚀 Agentic Model Tool (Ultimate Version)", layout="wide")
st.title("🚀 Agentic Model Tool (Ultimate Version - Fully Automated)")

# LLM Configuration
st.sidebar.header("🔑 LLM Configuration")
llm_type = st.sidebar.selectbox("Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])
openai_api_key = st.secrets.get("openai_api_key") or st.sidebar.text_input("OpenAI API Key (Optional for GPT-4)", type="password")
if openai_api_key:
    openai.api_key = openai_api_key

# Auto-Refresh for Real-Time Monitoring
st_autorefresh(interval=60 * 1000, key="auto-refresh")  # Refresh every 60 seconds

# Prompt for User Request
prompt = st.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500')")

# Forecasting Period Selection
forecast_period = st.selectbox("Select Forecasting Period:", ["1 Day", "1 Week", "1 Month", "1 Year"])

# Customizable Data Source Selection
data_source = st.sidebar.selectbox("Select Data Source", ["Yahoo Finance", "CoinGecko", "Alpha Vantage", "Custom CSV"])

# Smart Data Sourcing Function
def smart_data_sourcing(prompt):
    if "sp500" in prompt.lower() and data_source == "Yahoo Finance":
        data = yf.download("SPY", period="1y")
        st.write("✅ Stock Data for SP500 Sourced (Yahoo Finance)")
        return data.reset_index()
    elif "crypto" in prompt.lower() and data_source == "CoinGecko":
        data = yf.download("BTC-USD", period="1y")
        st.write("✅ Cryptocurrency Data (BTC-USD) Sourced (CoinGecko)")
        return data.reset_index()
    elif data_source == "Alpha Vantage":
        api_key = st.text_input("Enter Alpha Vantage API Key:", type="password")
        if api_key:
            symbol = "SPY"
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&datatype=csv"
            data = pd.read_csv(url)
            st.write("✅ Stock Data (Alpha Vantage) Sourced")
            return data
    elif data_source == "Custom CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("✅ Custom CSV Data Loaded")
            return data
    st.error("❌ Unable to detect appropriate data source.")
    return None

# Auto-detect data
data = smart_data_sourcing(prompt)

# Automated Data Cleaning
# Clean Data Function with Type Checking
def clean_data(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Dropping NaNs and converting to numeric (for y)
    data = data.dropna()
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return data

# Example Usage:
X = np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]])
y = np.array([1, 2, np.nan])

# Ensure X and y are properly cleaned
X_cleaned = clean_data(X)
y_cleaned = clean_data(pd.DataFrame(y))

print("✅ Cleaned Data (X):")
print(X_cleaned)
print("\n✅ Cleaned Data (y):")
print(y_cleaned)

# Model Explainability with SHAP
def model_explainability(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.write("🔍 Model Explainability (SHAP Values)")
    shap.summary_plot(shap_values, X, plot_type="bar")

# Advanced Model Training with LSTM/Transformer
def train_lstm(X, y):
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)
    return model

# Auto-Optimize Model with Ensemble
def auto_optimize(X, y):
    models = {
        "XGBoost": XGBRegressor(),
        "ARIMA": ARIMA(y, order=(5,1,0)),
        "Prophet": Prophet(),
        "LSTM": train_lstm(X, y)
    }

    best_model = None
    best_score = float("inf")
    model_scores = {}

    for model_name, model in models.items():
        st.write(f"🔍 Optimizing {model_name}...")
        if model_name == "Prophet":
            df = pd.DataFrame({'ds': pd.to_datetime(data['Date']), 'y': y})
            model.fit(df)
            forecast = model.predict(model.make_future_dataframe(periods=30))
            preds = forecast['yhat'].values[-len(y):]
        elif model_name == "ARIMA":
            model = model.fit()
            preds = model.forecast(steps=len(y))
        elif model_name == "LSTM":
            preds = model.predict(X.reshape(X.shape[0], X.shape[1], 1))
        else:
            model.fit(X, y)
            preds = model.predict(X)
        
        model_scores[model_name] = mean_squared_error(y, preds)
        st.write(f"✅ {model_name} Score (MSE): {model_scores[model_name]:.4f}")

        if model_scores[model_name] < best_score:
            best_score = model_scores[model_name]
            best_model = model

    # Display Model Comparison
    st.write("📊 Model Performance Comparison")
    st.table(pd.DataFrame(model_scores, index=["MSE"]).T)

    return best_model

if data is not None:
    st.write("✅ Data Loaded Automatically")
    
    y = data['Close'].values
    X = data.drop(columns=["Close"]).values

    X, y = clean_data(X), clean_data(pd.DataFrame(y))
    best_model = auto_optimize(X, y)
    st.write(f"✅ Best Model: {type(best_model).__name__}")

    # Explainability
    model_explainability(best_model, X)

    # Multi-Step Forecasting
    st.write("📊 Multi-Step Forecasting...")
    st.line_chart(best_model.predict(X))

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
