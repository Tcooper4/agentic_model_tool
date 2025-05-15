import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import optuna
import yfinance as yf
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from fpdf import FPDF

try:
    from statsmodels.tsa.arima.model import ARIMA
    arima_available = True
except ImportError:
    arima_available = False

try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False

st.title("🚀 Advanced Agentic Model Tool with Real-Time Forecasting + PDF Reports")
st.sidebar.header("🔧 Configuration")

# LLM Configuration
st.sidebar.subheader("🔑 LLM Configuration")
llm_type = st.sidebar.selectbox("Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])
if llm_type == "OpenAI (GPT-4)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Required for GPT-4)", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key
        st.sidebar.write("💡 Estimated Cost: ~$0.03 per 1,000 tokens")

# Prompt for User Request
prompt = st.text_input("Enter Your Request or Prompt (e.g., 'Predict SP500 for this week')")

# Enhanced Smart Data Sourcing Function
def smart_data_sourcing(prompt):
    prompt = prompt.lower()
    if "sp500" in prompt or "stock" in prompt or "ticker" in prompt:
        ticker = "SPY"
        data = yf.download(ticker, period="1y")
        st.write(f"✅ Stock Data for {ticker} (Yahoo Finance) Sourced")
        return data.reset_index()
    
    elif "crypto" in prompt or "bitcoin" in prompt or "ethereum" in prompt:
        crypto = "bitcoin"
        url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days=365"
        response = requests.get(url).json()
        prices = response.get('prices', [])
        if prices:
            data = pd.DataFrame(prices, columns=["timestamp", "price"]).set_index("timestamp")
            st.write("✅ Cryptocurrency Data (CoinGecko API) Sourced")
            return data
    
    st.error("❌ Unable to detect appropriate data source.")
    return None

# Auto-detect data based on prompt
data = smart_data_sourcing(prompt)
if data is not None and not data.empty:
    st.write("✅ Data Loaded Automatically")
    st.write(data.head())

    y = data.iloc[:, -1].values
    X = data.iloc[:, :-1].values

    # Auto-detect task type
    if len(set(y)) <= 20:
        model = XGBClassifier()
        task_type = "Classification"
    elif "forecast" in prompt or "predict" in prompt:
        task_type = "Forecasting"
    else:
        model = XGBRegressor()
        task_type = "Regression"
    
    st.write(f"🚀 Detected Task: {task_type}")

    # Data Preprocessing Function
    def preprocess_data(X, y):
        X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Ensure y is a Series (single column)
        if isinstance(y, pd.DataFrame) or isinstance(y, np.ndarray):
            y = pd.Series(y.flatten())  # Convert DataFrame/Array to Series
        
        if y.dtype == 'object':  # If y is not numeric
            y = pd.to_numeric(y, errors='coerce').fillna(0)
        
        # Handle classification vs. regression
        if len(set(y)) > 20:  # Regression
            y = y.astype(float)
        else:  # Classification
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        
        return X, y

    X, y = preprocess_data(X, y)

    # Model Training and Optimization
    def objective(trial):
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
    preds = model.predict(X)

    if task_type == "Classification":
        st.write(f"🔍 Accuracy: {accuracy_score(y, preds)}")
    else:
        st.write(f"🔍 Mean Squared Error: {mean_squared_error(y, preds)}")

    # PDF Report Generation Function
    def generate_pdf_report(data, model, preds, task_type):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Agentic Model Tool - Report", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Task Type: {task_type}")
        pdf.multi_cell(0, 10, f"Model Used: {type(model).__name__}")
        
        if task_type == "Classification":
            pdf.multi_cell(0, 10, f"Accuracy: {accuracy_score(y, preds)}")
        else:
            pdf.multi_cell(0, 10, f"Mean Squared Error: {mean_squared_error(y, preds)}")

        pdf.ln(10)
        pdf.multi_cell(0, 10, "Sample Data Preview:")
        for i in range(min(5, len(data))):
            pdf.multi_cell(0, 10, str(data.iloc[i].to_dict()))

        pdf.output("Agentic_Model_Report.pdf")
        st.success("✅ PDF Report Generated: Agentic_Model_Report.pdf")
        st.download_button("📥 Download PDF Report", data=open("Agentic_Model_Report.pdf", "rb"), file_name="Agentic_Model_Report.pdf")

    # PDF Report Button
    if st.button("Generate PDF Report"):
        generate_pdf_report(data, model, preds, task_type)

else:
    st.error("❌ No Data Available. Please enter a prompt or upload a file.")
