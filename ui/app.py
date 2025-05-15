
import streamlit as st
import yaml
from core.model_factory import create_model

st.title("Agentic Model Creation Tool")

st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox("Choose Model Type", ["XGBoost", "LSTM", "Linear Regression", "Transformer"])

st.sidebar.subheader("Hyperparameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
max_depth = st.sidebar.slider("Max Depth (XGBoost Only)", 1, 20, 6)
n_estimators = st.sidebar.slider("Number of Estimators (XGBoost Only)", 10, 500, 100)

if st.button("Create Model"):
    config = {
        "model_type": model_type,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "max_depth": max_depth if model_type == "XGBoost" else None,
            "n_estimators": n_estimators if model_type == "XGBoost" else None,
        }
    }
    with open("config/user_model.yaml", "w") as file:
        yaml.dump(config, file)
    
    st.success(f"Model configuration saved as user_model.yaml. You can now train it.")

if st.button("Train Model"):
    with open("config/user_model.yaml", "r") as file:
        config = yaml.safe_load(file)
    model = create_model(config)
    st.write("Model Created:", model)

st.subheader("Test Model")
uploaded_file = st.file_uploader("Upload a CSV file for testing")
if uploaded_file:
    import pandas as pd
    data = pd.read_csv(uploaded_file)
    st.write("Data Sample:", data.head())
