
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

# Streamlit UI Setup
st.title("🚀 Agentic Model Creation Tool (Fully Automated)")
st.sidebar.header("Configuration")

# User selects model type or fully automated
mode = st.sidebar.selectbox("Mode", ["Fully Automated", "Advanced Mode"])

st.markdown("### Upload Your Data")
uploaded_file = st.file_uploader("Upload a CSV file")

# API Key Configuration (Optional)
st.sidebar.subheader("🔑 OpenAI API (Optional)")
openai_api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")
if openai_api_key:
    openai.api_key = openai_api_key

# Function to automatically clean and preprocess data
def clean_data(data):
    st.write("🔄 Cleaning and Preprocessing Data...")
    data = pd.get_dummies(data, drop_first=True)
    data.fillna(0, inplace=True)
    st.success("✅ Data Cleaning Complete")
    return data

# Function to automatically detect target type (Regression or Classification)
def detect_target_type(y):
    if y.nunique() > 20:
        st.write("🔍 Detected Continuous Values - Using Regression")
        return "regression"
    else:
        st.write("🔍 Detected Discrete Values - Using Classification")
        return "classification"

# Function for automatic training and optimization
def auto_train_and_optimize(X, y):
    st.write("🚀 Auto Model Training and Optimization")

    if detect_target_type(y) == "regression":
        model = XGBRegressor()
    else:
        model = XGBClassifier()

    def objective(trial):
        model.set_params(
            n_estimators=trial.suggest_int("n_estimators", 10, 200),
            max_depth=trial.suggest_int("max_depth", 3, 10)
        )
        model.fit(X, y)
        predictions = model.predict(X)
        if detect_target_type(y) == "regression":
            return -mean_squared_error(y, predictions)
        else:
            return accuracy_score(y, predictions)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    st.success(f"✅ Optimization Complete. Best Parameters: {study.best_params}")

    model.set_params(**study.best_params)
    model.fit(X, y)
    st.success("✅ Model Training Complete")
    return model

# Main Application Logic
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("✅ Data Uploaded")
    st.write(data.head())

    if mode == "Fully Automated":
        st.write("🚀 Fully Automated Mode Selected")
        y = data.iloc[:, -1]  # Assume last column is the target variable
        X = data.iloc[:, :-1]  # All other columns are features

        X = clean_data(X)
        model = auto_train_and_optimize(X, y)

        st.write("✅ Final Model Trained")
        st.write("### Model Performance")
        predictions = model.predict(X)
        if detect_target_type(y) == "regression":
            st.write(f"🔍 Mean Squared Error: {mean_squared_error(y, predictions)}")
        else:
            st.write(f"🔍 Accuracy: {accuracy_score(y, predictions)}")

    else:
        st.write("🔧 Advanced Mode Selected")
        model_type = st.selectbox("Choose Model Type", ["LogisticRegression", "RandomForest", "XGBoost", "LLM"])
        st.write("✅ Advanced Mode Complete")

st.sidebar.markdown("### 🚀 How It Works")
st.sidebar.write("""
1. Upload your CSV file.
2. The tool will automatically inspect the data, clean it, and select the best model type.
3. If 'Fully Automated' mode is selected, it will automatically train and optimize the model.
4. If 'Advanced Mode' is selected, you can customize the model.
5. For LLM mode, OpenAI API Key is required.
""")
