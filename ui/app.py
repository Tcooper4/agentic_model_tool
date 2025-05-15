
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
import os
import requests

st.title("🚀 Agentic Model Creation Tool (Fully Automated with Data Sourcing)")
st.sidebar.header("Configuration")

# User selects mode
mode = st.sidebar.selectbox("Mode", ["Fully Automated", "Advanced Mode"])
data_source = st.sidebar.selectbox("Data Source", ["Upload CSV", "Stock Data (Yahoo Finance)", "Crypto (CoinGecko)", "Economic Data (FRED)"])

prompt = st.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500', 'Classify Emails')")

# API Key Configuration (Optional)
st.sidebar.subheader("🔑 OpenAI API (Optional)")
openai_api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")
if openai_api_key:
    openai.api_key = openai_api_key
    st.sidebar.write("💡 Estimated Cost: ~$0.03 per 1,000 tokens")

def fetch_api_data(source):
    if source == "Stock Data (Yahoo Finance)":
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)")
        if ticker:
            data = yf.download(ticker, period="1y")
            st.write(data.head())
            return data.reset_index()

    elif source == "Crypto (CoinGecko)":
        crypto = st.text_input("Enter Cryptocurrency (e.g., bitcoin, ethereum)")
        if crypto:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days=365"
            response = requests.get(url).json()
            prices = response['prices']
            data = pd.DataFrame(prices, columns=["timestamp", "price"]).set_index("timestamp")
            st.write(data.head())
            return data

    elif source == "Economic Data (FRED)":
        indicator = st.text_input("Enter FRED Indicator (e.g., GDP, CPI)")
        if indicator:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={indicator}&api_key=YOUR_FRED_API_KEY&file_type=json"
            response = requests.get(url).json()
            observations = response['observations']
            data = pd.DataFrame(observations)
            st.write(data.head())
            return data

    return None

# Main Application Logic
data = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("✅ Data Uploaded")
        st.write(data.head())
else:
    data = fetch_api_data(data_source)

# If data is available, continue with model training
if data is not None and not data.empty:
    st.write("✅ Data Ready")

    if mode == "Fully Automated":
        st.write("🚀 Fully Automated Mode Selected")
        y = data.iloc[:, -1]  # Assume last column is the target variable
        X = data.iloc[:, :-1]  # All other columns are features

        model = XGBClassifier() if y.nunique() <= 20 else XGBRegressor()
        model.fit(X, y)
        st.write("✅ Model Trained Automatically")
        st.write("### Model Performance")
        predictions = model.predict(X)
        if y.nunique() > 20:
            st.write(f"🔍 Mean Squared Error: {mean_squared_error(y, predictions)}")
        else:
            st.write(f"🔍 Accuracy: {accuracy_score(y, predictions)}")

    else:
        st.write("🔧 Advanced Mode Selected")
        model_type = st.selectbox("Choose Model Type", ["LogisticRegression", "RandomForest", "XGBoost", "LLM"])

        if model_type == "LLM":
            if openai_api_key:
                response = openai.Completion.create(
                    engine="gpt-4",
                    prompt=prompt,
                    max_tokens=100,
                    n=1
                )
                st.write("🔮 LLM Response:")
                st.write(response.choices[0].text.strip())
            else:
                st.error("❌ Please enter your OpenAI API Key for LLM")

        else:
            st.write("🔍 Training Custom Model")
            y = data.iloc[:, -1]  # Assume last column is the target variable
            X = data.iloc[:, :-1]
            if model_type == "LogisticRegression":
                model = LogisticRegression()
            elif model_type == "RandomForest":
                model = RandomForestClassifier()
            else:
                model = XGBClassifier() if y.nunique() <= 20 else XGBRegressor()

            model.fit(X, y)
            st.write("✅ Model Trained")
            st.write("### Model Performance")
            predictions = model.predict(X)
            if y.nunique() > 20:
                st.write(f"🔍 Mean Squared Error: {mean_squared_error(y, predictions)}")
            else:
                st.write(f"🔍 Accuracy: {accuracy_score(y, predictions)}")
else:
    st.error("❌ No Data Available. Please upload a file or connect to an API.")

st.sidebar.markdown("### 🚀 How It Works")
st.sidebar.write("""
1. Choose data source (Upload CSV, Stock, Crypto, or Economic Data).
2. If using API data, enter the specific symbol (e.g., AAPL for stocks).
3. Choose 'Fully Automated' for automatic model selection or 'Advanced' to customize.
4. If 'LLM' is chosen, enter an OpenAI API Key for GPT-4.
5. Click 'Create and Train Model' to start.
""")
