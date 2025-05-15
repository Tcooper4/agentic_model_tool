
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

st.title("🚀 Agentic Model Creation Tool (Fully Automated)")
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

# Smart Data Sourcing with Auto-Correction and Suggestions
def smart_data_sourcing(prompt):
    prompt = prompt.lower()
    
    if "sp500" in prompt or "stock" in prompt or "ticker" in prompt:
        ticker = "SPY"
        try:
            data = yf.download(ticker, period="1y")
            if not data.empty:
                st.write(f"✅ Stock Data for {ticker} (Yahoo Finance) Sourced")
                return data.reset_index()
        except Exception as e:
            st.error(f"❌ Error fetching stock data: {str(e)}")
    
    elif "crypto" in prompt or "bitcoin" in prompt or "ethereum" in prompt:
        crypto = "bitcoin"
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days=365"
            response = requests.get(url).json()
            prices = response.get('prices', [])
            if prices:
                data = pd.DataFrame(prices, columns=["timestamp", "price"]).set_index("timestamp")
                st.write("✅ Cryptocurrency Data (CoinGecko API) Sourced")
                return data
        except Exception as e:
            st.error(f"❌ Error fetching crypto data: {str(e)}")
    
    elif "gdp" in prompt or "cpi" in prompt or "unemployment" in prompt:
        st.error("❌ Economic data (FRED) is currently not supported automatically.")
    
    st.error("❌ Unable to detect appropriate data source. Please enter a valid request.")
    return None

# Auto-detect data based on prompt
data = smart_data_sourcing(prompt)

if data is not None and not data.empty:
    st.write("✅ Data Ready")
    
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    # Auto-detect task type and model selection
    if y.nunique() <= 20:
        model = XGBClassifier()
        task_type = "Classification"
    else:
        model = XGBRegressor()
        task_type = "Regression"
    
    st.write(f"🚀 Auto-detected task: {task_type}")

    # Auto-optimize model using Optuna
    def objective(trial):
        if task_type == "Classification":
            model.n_estimators = trial.suggest_int("n_estimators", 50, 500)
            model.max_depth = trial.suggest_int("max_depth", 2, 10)
            model.learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        else:
            model.n_estimators = trial.suggest_int("n_estimators", 50, 500)
            model.max_depth = trial.suggest_int("max_depth", 2, 10)
            model.learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        
        model.fit(X, y)
        preds = model.predict(X)
        if task_type == "Classification":
            return 1 - accuracy_score(y, preds)
        else:
            return mean_squared_error(y, preds)
    
    st.write("🚀 Auto-Optimizing Model...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    st.write("✅ Model Optimized Automatically")

    # Display Results
    preds = model.predict(X)
    if task_type == "Classification":
        st.write(f"🔍 Accuracy: {accuracy_score(y, preds)}")
    else:
        st.write(f"🔍 Mean Squared Error: {mean_squared_error(y, preds)}")
else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
