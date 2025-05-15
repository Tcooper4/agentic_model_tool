import subprocess
import sys
import streamlit as st
import pandas as pd
import openai
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ✅ Auto-Detecting and Installing Compatible Torch Version (for Python 3.12)
def ensure_torch():
    try:
        import torch
        st.write(f"Torch version {torch.__version__} is already installed.")
    except ImportError:
        st.warning("Torch not detected. Auto-installing the correct version for CPU...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", 
            "--index-url", "https://download.pytorch.org/whl/torch_stable.html"
        ], check=True)
        import torch
        st.success(f"Torch version {torch.__version__} installed successfully.")

# ✅ Ensure Torch is installed
ensure_torch()

# ✅ Standard App Code Below (LLM Model Creation Tool)
st.title("Autonomous Agentic Model Creation Tool (Secure LLM Choice)")

st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox("Choose Model Type", ["LogisticRegression", "RandomForest", "XGBoost", "LLM"])
llm_type = st.sidebar.selectbox("LLM Type", ["Hugging Face (Free)", "OpenAI (GPT-4)"])

# ✅ Secure API Key Input (Only stored in session)
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
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            st.success(f"LLM Model (Hugging Face - {model_name}) created successfully.")
        
        elif llm_type == "OpenAI (GPT-4)" and "openai_api_key" in st.session_state:
            openai.api_key = st.session_state["openai_api_key"]
            st.success("LLM Model (OpenAI GPT-4) configured. Ready for classification or generation.")
        
        else:
            st.error("Please enter your OpenAI API Key for GPT-4.")

    if model and uploaded_file:
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
                st.write("Hugging Face model loaded. Fine-tuning skipped for simplicity.")
                # Optionally, you can add your Hugging Face classification logic here

            data['Predictions'] = predictions
            st.write("Classification Results:", data[[text_column, 'Predictions']])
            st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv")

        elif task_type == "generation":
            texts = data[text_column].astype(str).tolist()
            generated_texts = []

            if llm_type == "OpenAI (GPT-4)" and "openai_api_key" in st.session_state:
                for text in texts:
                    response = openai.Completion.create(
                        engine="gpt-4",
                        prompt=f"Generate text based on: {text}",
                        max_tokens=100,
                        n=1,
                        temperature=0.7
                    )
                    generated_texts.append(response.choices[0].text.strip())

            data['Generated Text'] = generated_texts
            st.write("Generated Text Results:", data[[text_column, 'Generated Text']])
            st.download_button("Download Generated Text", data.to_csv(index=False), "generated_text.csv")
