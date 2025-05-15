import streamlit as st
import pandas as pd
import openai
import importlib.util
import subprocess
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import optuna
import numpy as np
import re

# ✅ Auto-Install Missing Libraries (yfinance, etc.)
def ensure_package_installed(package):
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

ensure_package_installed("yfinance")
import yfinance as yf

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
        try:
            # ✅ Import XGBoost Classifier and Regressor at the start
            from xgboost import XGBClassifier, XGBRegressor

            # ✅ Ensure X_train is a DataFrame and y_train is a Series
            X_train = pd.DataFrame(X_train)
            y_train = pd.Series(y_train)

            # ✅ Convert Categorical Data to Numerical (if any)
            X_train = pd.get_dummies(X_train, drop_first=True)

            # ✅ Handle Missing Values
            X_train.fillna(0, inplace=True)
            y_train.fillna(0, inplace=True)

            # ✅ Ensure all data is numeric
            X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)

            # ✅ Detect if target is continuous (Regression) or discrete (Classification)
            if self.model_type == "XGBoost":
                if y_train.nunique() > 20:
                    st.warning("⚠️ Detected continuous values in the target variable. Switching to XGBRegressor.")
                    self.model = XGBRegressor()
                else:
                    st.success("✅ Detected discrete classes. Using XGBClassifier.")
                    self.model = XGBClassifier()

            # ✅ Convert Continuous Values to Classes if Classification
            if self.model_type == "XGBoost" and isinstance(self.model, XGBClassifier):
                if y_train.nunique() > 20:
                    st.warning("⚠️ Target values are continuous. Converting to categories for classification.")
                    y_train = pd.cut(y_train, bins=20, labels=False)
            
            if self.agentic_mode:
                def objective(trial):
                    if self.model_type == "LogisticRegression":
                        self.model = LogisticRegression(C=trial.suggest_loguniform("C", 0.01, 10))
                    elif self.model_type == "RandomForest":
                        self.model = RandomForestClassifier(
                            n_estimators=trial.suggest_int("n_estimators", 10, 200)
                        )
                    elif self.model_type == "XGBoost":
                        if isinstance(self.model, XGBClassifier):
                            self.model = XGBClassifier(
                                n_estimators=trial.suggest_int("n_estimators", 10, 200),
                                max_depth=trial.suggest_int("max_depth", 3, 10),
                            )
                        else:
                            self.model = XGBRegressor(
                                n_estimators=trial.suggest_int("n_estimators", 10, 200),
                                max_depth=trial.suggest_int("max_depth", 3, 10),
                            )
                    self.model.fit(X_train, y_train)
                    predictions = self.model.predict(X_train)
                    
                    # ✅ Use appropriate metric for classification or regression
                    if isinstance(self.model, XGBClassifier):
                        return accuracy_score(y_train, predictions)
                    else:
                        return -np.mean((y_train - predictions) ** 2)
                
                st.write("🚀 Optimizing model with Optuna (Auto-Optimization)...")
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=10)
                self.best_params = study.best_params
                st.success(f"✅ Optimization complete. Best parameters: {self.best_params}")
                self.model.set_params(**self.best_params)
            else:
                self.model.fit(X_train, y_train)

        except ValueError as ve:
            st.error(f"❌ XGBoost encountered an error: {str(ve)}")
            st.error("⚠️ This is often caused by non-numerical data, missing values, or continuous target values in classification.")

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        st.write(f"✅ Model Performance - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

        if self.model_type in ["RandomForest", "XGBoost"]:
            st.write("📊 Feature Importance:")
            if self.model_type == "XGBoost":
                plot_importance(self.model)
                st.pyplot(plt)
            else:
                feature_importance = pd.Series(self.model.feature_importances_, index=X_test.columns)
                st.bar_chart(feature_importance.sort_values(ascending=False))

# ✅ Streamlit UI (Highly User-Friendly)
st.title("🌐 Agentic Model Creation Tool (Natural Language Prompting + LLM API Choice)")

st.markdown("""
### Welcome to the Agentic Model Tool! 🚀
- Simply enter what you want to do using plain language.
- The tool will automatically understand your instructions and build the best model for you.
- Choose between **OpenAI GPT-4 (Paid)** and **Hugging Face (Free)** for LLM tasks.
""")

# ✅ LLM API Choice
llm_api_choice = st.sidebar.selectbox("Choose LLM API:", ["OpenAI GPT-4 (Paid)", "Hugging Face (Free - CPU Only)"])

if llm_api_choice == "OpenAI GPT-4 (Paid)":
    st.sidebar.write("🔑 **API Key Required for GPT-4**")
    openai_api_key = st.sidebar.text_input("Enter Your OpenAI API Key (Secure)", type="password")
    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key
    st.sidebar.write("💡 **Estimated Cost:** ~$0.03 per 1000 tokens (GPT-4)")

prompt_text = st.text_area("Enter Your Prompt (e.g., 'Forecast AAPL stock price.')", placeholder="Type your instructions here...")

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
        data['Return'] = data['Close'].pct_change().dropna()
        X = np.array(range(len(data))).reshape(-1, 1)
        y = data['Return'].dropna()
        agent = AgenticModel(model_type="XGBoost", agentic_mode=True)
        agent.create_model()
        agent.train_and_optimize(X, y)
elif task_type == "classification":
    uploaded_file = st.file_uploader("📂 Upload a CSV file for training (Required)")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        target_column = st.selectbox("Select Target Column (Label):", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        agent = AgenticModel(model_type="RandomForest", agentic_mode=True)
        agent.create_model()
        agent.train_and_optimize(X, y)
elif task_type == "text-generation" and llm_api_choice == "OpenAI GPT-4 (Paid)":
    if "openai_api_key" in st.session_state:
        openai.api_key = st.session_state["openai_api_key"]
        response = openai.Completion.create(engine="gpt-4", prompt=prompt_text, max_tokens=100, temperature=0.7)
        st.write("✅ Generated Text:", response.choices[0].text.strip())
