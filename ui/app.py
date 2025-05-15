
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import optuna
import yfinance as yf
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests

st.title("🚀 Agentic Model Creation Tool (Fully Automated with Intelligent Data Sourcing)")
st.sidebar.header("Configuration")

# LLM Configuration (Choose between Hugging Face or OpenAI)
st.sidebar.subheader("🔑 LLM Configuration")
llm_type = st.sidebar.selectbox("Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])

if llm_type == "OpenAI (GPT-4)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Required for GPT-4)", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key
        st.sidebar.write("💡 Estimated Cost: ~$0.03 per 1,000 tokens")

# Prompt for User Request
prompt = st.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500', 'Classify Emails')")

# Intelligent Data Sourcing
def intelligent_data_sourcing(prompt):
    if "SP500" in prompt or "stock" in prompt or "ticker" in prompt:
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", value="SPY")
        if ticker:
            data = yf.download(ticker, period="1y")
            st.write("✅ Stock Data (Yahoo Finance) Sourced")
            return data.reset_index()
    
    elif "crypto" in prompt or "bitcoin" in prompt or "ethereum" in prompt:
        crypto = st.text_input("Enter Cryptocurrency (e.g., bitcoin, ethereum)", value="bitcoin")
        if crypto:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days=365"
            response = requests.get(url).json()
            prices = response['prices']
            data = pd.DataFrame(prices, columns=["timestamp", "price"]).set_index("timestamp")
            st.write("✅ Cryptocurrency Data (CoinGecko API) Sourced")
            return data
    
    elif "GDP" in prompt or "CPI" in prompt or "unemployment" in prompt:
        indicator = st.text_input("Enter FRED Indicator (e.g., GDP, CPI)", value="GDP")
        if indicator:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={indicator}&api_key=YOUR_FRED_API_KEY&file_type=json"
            response = requests.get(url).json()
            observations = response['observations']
            data = pd.DataFrame(observations)
            st.write("✅ Economic Data (FRED API) Sourced")
            return data
    
    st.error("❌ Unable to detect appropriate data source. Please upload a CSV.")
    return None

# Auto-detect data based on prompt
data = intelligent_data_sourcing(prompt)

if data is not None and not data.empty:
    st.write("✅ Data Ready")
    
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    # Automatically choose model based on target type
    model = XGBClassifier() if y.nunique() <= 20 else XGBRegressor()
    model.fit(X, y)
    st.write("✅ Model Trained Automatically")
    st.write("### Model Performance")
    predictions = model.predict(X)
    st.write(f"🔍 Accuracy: {accuracy_score(y, predictions)}") if y.nunique() <= 20 else st.write(f"🔍 Mean Squared Error: {mean_squared_error(y, predictions)}")

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
