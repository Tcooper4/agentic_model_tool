
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import yaml
from core.model_factory import create_model
from core.performance_evaluator import evaluate_model
from core.optimizer import optimize_model

st.title("Autonomous Agentic Model Creation Tool")

st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox("Choose Model Type", ["LogisticRegression", "RandomForest", "XGBoost", "LLM"])
task_type = st.sidebar.selectbox("Task Type", ["classification", "regression"])

st.sidebar.subheader("Training Data")
uploaded_file = st.file_uploader("Upload a CSV file for training")

if st.button("Create and Train Model"):
    config = {"model_type": model_type}
    model = create_model(config)
    st.success(f"Model created: {model}")

    if uploaded_file:
        import pandas as pd
        data = pd.read_csv(uploaded_file)
        X = data.drop(columns="target")
        y = data["target"]
        
        # Training Model
        model.fit(X, y)
        st.write("Model Trained Successfully")

        # Evaluating Model
        metrics = evaluate_model(model, X, y, task_type)
        st.write("Performance:", metrics)

        # Optimizing Model
        st.write("Optimizing Model...")
        model = optimize_model(model, X, y)
        st.success("Model Optimized")
