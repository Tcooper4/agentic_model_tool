import streamlit as st
import pandas as pd
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna
import numpy as np

# ✅ Agentic System (Dynamic Model Creation, Evaluation, and Optimization)
class AgenticModel:
    def __init__(self, model_type, llm_type=None, agentic_mode=False):
        self.model_type = model_type
        self.llm_type = llm_type
        self.model = None
        self.best_params = None
        self.agentic_mode = agentic_mode

    def create_model(self):
        if self.model_type == "LogisticRegression":
            self.model = LogisticRegression()
        elif self.model_type == "RandomForest":
            self.model = RandomForestClassifier()
        elif self.model_type == "XGBoost":
            self.model = XGBClassifier()
        elif self.model_type == "LLM" and self.llm_type == "Hugging Face (Free - CPU Only)":
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        st.success(f"✅ {self.model_type} model created.")

    def run_llm_prompt(self, prompt_text, temperature, max_tokens, top_p, freq_penalty):
        if self.llm_type == "Hugging Face (Free - CPU Only)":
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            outputs = self.model(**inputs)
            return outputs.logits.argmax().item()

        elif self.llm_type == "OpenAI (GPT-4)" and "openai_api_key" in st.session_state:
            response = openai.Completion.create(
                engine="gpt-4",
                prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=freq_penalty,
                n=1
            )
            return response.choices[0].text.strip()

    def train_and_optimize(self, X_train, y_train):
        if self.model_type in ["LogisticRegression", "RandomForest", "XGBoost"]:
            if self.agentic_mode:
                def objective(trial):
                    if self.model_type == "LogisticRegression":
                        self.model = LogisticRegression(C=trial.suggest_loguniform("C", 0.01, 10))
                    elif self.model_type == "RandomForest":
                        self.model = RandomForestClassifier(
                            n_estimators=trial.suggest_int("n_estimators", 10, 200)
                        )
                    elif self.model_type == "XGBoost":
                        self.model = XGBClassifier(
                            n_estimators=trial.suggest_int("n_estimators", 10, 200),
                            max_depth=trial.suggest_int("max_depth", 3, 10),
                        )
                    self.model.fit(X_train, y_train)
                    predictions = self.model.predict(X_train)
                    return accuracy_score(y_train, predictions)
                
                st.write("🚀 Optimizing model with Optuna (Auto-Optimization)...")
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=10)
                self.best_params = study.best_params
                st.success(f"✅ Optimization complete. Best parameters: {self.best_params}")
                self.model.set_params(**self.best_params)
            else:
                self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        if self.model_type in ["LogisticRegression", "RandomForest", "XGBoost"]:
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            st.write(f"✅ Model Performance - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

# ✅ Streamlit UI (Highly User-Friendly)
st.title("🌐 Agentic Model Creation Tool (Beginner-Friendly)")

st.markdown("""
### Welcome to the Agentic Model Tool! 🚀
This tool allows you to easily create, optimize, and use machine learning models:
- **Logistic Regression, Random Forest, XGBoost:** For data classification.
- **LLM (GPT-4, Hugging Face):** For text generation or classification.
- **Agentic Mode:** Automatically optimizes models for the best performance.
""")

st.sidebar.header("🔧 Model Configuration")
model_type = st.sidebar.selectbox("Choose Model Type", ["LogisticRegression", "RandomForest", "XGBoost", "LLM"])
llm_type = st.sidebar.selectbox("LLM Type", ["Hugging Face (Free - CPU Only)", "OpenAI (GPT-4)"])

# ✅ Secure API Key Input (Only stored in session)
if llm_type == "OpenAI (GPT-4)":
    st.sidebar.write("🔑 **API Key Required for GPT-4**")
    openai_api_key = st.sidebar.text_input("Enter Your OpenAI API Key (Secure)", type="password")
    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key
    st.sidebar.write("💡 **Estimated Cost:** Each request costs ~$0.03 per 1000 tokens for GPT-4.")

# ✅ Agentic Mode Toggle
agentic_mode = st.sidebar.checkbox("Enable Agentic Mode (Auto-Optimization)", value=True)
st.sidebar.write("""
**Agentic Mode:** 
- Automatically finds the best settings for your model using Optuna.
- Only affects non-LLM models (Logistic Regression, Random Forest, XGBoost).
""")

# ✅ LLM Settings (Detailed Descriptions)
if model_type == "LLM":
    st.markdown("### 📌 LLM Prompt Settings (Text Generation)")
    prompt_text = st.text_area("Enter your prompt for the LLM", placeholder="Type your prompt here...")
    
    st.write("""
    **Temperature:** Controls creativity. Higher values (0.8 - 1.0) make output more random. 
    Lower values (0.2 - 0.5) make output more focused.
    """)
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7)
    
    st.write("**Max Tokens:** Maximum length of the generated text.")
    max_tokens = st.slider("Max Tokens (Response Length)", 10, 500, 100)
    
    st.write("**Top-P (Nucleus Sampling):** Controls diversity. Lower values make output more focused.")
    top_p = st.slider("Top-P (Diversity Control)", 0.0, 1.0, 0.9)
    
    st.write("**Frequency Penalty:** Prevents repetitive text in responses.")
    freq_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0)

uploaded_file = st.file_uploader("📂 Upload a CSV file for training (Optional for LLM)")

if st.button("🚀 Create and Train Model"):
    agent = AgenticModel(model_type=model_type, llm_type=llm_type, agentic_mode=agentic_mode)
    agent.create_model()
