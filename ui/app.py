import subprocess
import sys
import streamlit as st
import os

# Auto-Detect and Install the Correct Torch Version (Universal)
def ensure_torch():
    try:
        import torch
        st.write(f"Torch version {torch.__version__} is already installed.")
    except ImportError:
        st.warning("Torch not detected. Auto-installing the correct version...")

        python_version = sys.version_info
        if python_version >= (3, 12):
            # For Python 3.12+ (Streamlit Cloud)
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/nightly/cpu"])
        elif python_version >= (3, 8):
            # For Python 3.8 to 3.11
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
        else:
            st.error("Python version too old for Torch. Please upgrade.")

        try:
            import torch
            st.success(f"Torch version {torch.__version__} installed successfully.")
        except ImportError:
            st.error("Failed to install Torch. Please try restarting the app.")

# Ensure Torch is installed
ensure_torch()

# Proceed with your normal app code
st.title("Autonomous Agentic Model Creation Tool (Secure LLM Choice)")
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox("Choose Model Type", ["LogisticRegression", "RandomForest", "XGBoost", "LLM"])
llm_type = st.sidebar.selectbox("LLM Type", ["Hugging Face (Free)", "OpenAI (GPT-4)"])

# Secure API Key Input (Only stored in session)
if llm_type == "OpenAI (GPT-4)":
    openai_api_key = st.sidebar.text_input("Enter Your OpenAI API Key (Secure)", type="password")
    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key

task_type = st.sidebar.selectbox("Task Type", ["classification", "generation"])
uploaded_file = st.file_uploader("Upload a CSV file for training")

if st.button("Create and Train Model"):
    model = None
    if model_type == "LLM":
        if llm_type == "Hugging Face (Free)":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            st.success(f"LLM Model (Hugging Face - {model_name}) created successfully.")
        
        elif llm_type == "OpenAI (GPT-4)" and "openai_api_key" in st.session_state:
            import openai
            openai.api_key = st.session_state["openai_api_key"]
            st.success("LLM Model (OpenAI GPT-4) configured. Ready for classification or generation.")
        
        else:
            st.error("Please enter your OpenAI API Key for GPT-4.")

    if model and uploaded_file:
        import pandas as pd
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Sample:", data.head())
        
        text_column = st.selectbox("Select the Text Column for LLM:", data.columns)
        
        if task_type == "classification":
            texts = data[text_column].astype(str).tolist()
            predictions = []

            if llm_type == "OpenAI (GPT-4)" and "openai_api_key" in st.session_state:
                for text in texts:
                    response = openai.Completion.create(
                        engine="gpt-4",
                        prompt=f"Classify the following text: {text}",
                        max_tokens=50,
                        n=1,
                        temperature=0.7
                    )
                    predictions.append(response.choices[0].text.strip())

            elif llm_type == "Hugging Face (Free)":
                st.write("Hugging Face model loaded. Predictions can be added here.")

            data['Predictions'] = predictions
            st.write("Classification Results:", data[['text', 'Predictions']])
            st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv")

