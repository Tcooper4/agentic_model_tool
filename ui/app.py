import streamlit as st
import pandas as pd
import openai
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna
import numpy as np
import re

# ✅ Agentic System (Dynamic Model Creation, Evaluation, and Optimization)
class AgenticModel:
    def __init__(self, model_type, agentic_mode=False):
        self.model_type = model_type
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
        st.success(f"✅ {self.model_type} model created.")

    def train_and_optimize(self, X_train, y_train):
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
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        st.write(f"✅ Model Performance - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

# ✅ Streamlit UI (Highly User-Friendly)
st.title("🌐 Agentic Model Creation Tool (Natural Language Prompting)")

st.markdown("""
### Welcome to the Agentic Model Tool! 🚀
- Simply enter what you want to do using plain language.
- The tool will automatically understand your instructions and build the best model for you.
""")

prompt_text = st.text_area("Enter Your Prompt (e.g., 'Forecast AAPL stock price.')", placeholder="Type your instructions here...")

# ✅ Natural Language Understanding (Prompt Analysis)
def analyze_prompt(prompt):
    prompt = prompt.lower()
    if "forecast" in prompt or "predict" in prompt:
        return "forecasting"
    elif "classify" in prompt or "classification" in prompt:
        return "classification"
    elif "generate text" in prompt or "complete text" in prompt:
        return "text-generation"
    else:
        return "classification"

task_type = analyze_prompt(prompt_text)
st.write(f"🔍 Detected Task Type: **{task_type.capitalize()}**")

if task_type == "forecasting":
    ticker = re.search(r"\b[a-zA-Z]{1,5}\b", prompt_text)
    if ticker:
        ticker = ticker.group(0).upper()
        st.write(f"📈 Forecasting for ticker: {ticker}")
        data = yf.download(ticker, period="1y")
        st.write("📊 Stock Data Sample:", data.head())
        data['Return'] = data['Close'].pct_change().dropna()
        X = np.array(range(len(data))).reshape(-1, 1)
        y = data['Return'].dropna()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        agent = AgenticModel(model_type="XGBoost", agentic_mode=True)
        agent.create_model()
        agent.train_and_optimize(X_train, y_train)
        agent.evaluate_model(X_test, y_test)

elif task_type == "classification":
    st.write("📊 Classification Task Detected.")
    uploaded_file = st.file_uploader("📂 Upload a CSV file for training (Required)")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Sample:", data.head())
        target_column = st.selectbox("Select Target Column (Label):", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        agent = AgenticModel(model_type="RandomForest", agentic_mode=True)
        agent.create_model()
        agent.train_and_optimize(X_train, y_train)
        agent.evaluate_model(X_test, y_test)

elif task_type == "text-generation":
    st.write("📜 Text Generation Task Detected (LLM).")
    llm_type = st.selectbox("Choose LLM Type", ["Hugging Face (Free - CPU Only)", "OpenAI (GPT-4)"])
    
    if llm_type == "OpenAI (GPT-4)":
        openai_api_key = st.text_input("Enter Your OpenAI API Key (Secure)", type="password")
        if openai_api_key:
            st.session_state["openai_api_key"] = openai_api_key

    if st.button("🚀 Generate Text"):
        if llm_type == "OpenAI (GPT-4)" and "openai_api_key" in st.session_state:
            openai.api_key = st.session_state["openai_api_key"]
            response = openai.Completion.create(
                engine="gpt-4",
                prompt=prompt_text,
                max_tokens=100,
                temperature=0.7
            )
            st.write("✅ Generated Text:", response.choices[0].text.strip())

        elif llm_type == "Hugging Face (Free - CPU Only)":
            st.write("✅ Hugging Face text generation is not available in this setup.")
