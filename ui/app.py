import streamlit as st
import pandas as pd
import numpy as np
import openai
from transformers import pipeline
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, MultiHeadAttention, Input
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import traceback

# ✅ Streamlit Multi-Page Agentic Model Creation Tool (Prompt-Driven + LLM-Powered + Auto-Debug)
st.set_page_config(page_title="🚀 Fully Agentic Model Tool", layout="wide")
st.title("🚀 Fully Agentic Model Creation Tool (Prompt-Driven + LLM-Powered + Auto-Debug)")

# Sidebar Configuration (LLM Selection)
llm_type = st.sidebar.selectbox("🔑 Choose LLM", ["Hugging Face (Free)", "OpenAI (GPT-4)"])
openai_api_key = None

if llm_type == "OpenAI (GPT-4)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Required for GPT-4)", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key

# Sidebar Navigation (Pages)
page = st.sidebar.selectbox("Select Page", ["Home", "Forecasting", "Strategies", "Hybrid Model", "Backtesting"])

# ✅ Global Data Load (Simulated for Testing)
@st.cache_data(ttl=60 * 60)
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2015-01-01", periods=2000)
    prices = np.cumsum(np.random.randn(2000)) + 100
    return pd.DataFrame({'Date': dates, 'Target': prices})

data = load_data()
prompt = None

# ✅ Home Page (Prompt Input with LLM)
if page == "Home":
    st.header("Welcome to the Fully Agentic Model Tool")
    prompt = st.text_input("Enter Your Request (e.g., 'Predict SP500 and optimize strategies')", "")
    if st.button("Submit Request"):
        if llm_type == "OpenAI (GPT-4)" and openai_api_key:
            response = openai.Completion.create(
                model="gpt-4",
                prompt=f"Analyze this request: '{prompt}'. List actions to perform (forecast, strategy, hybrid, backtest).",
                max_tokens=50,
                temperature=0.2
            )
            actions = response.choices[0].text.strip().lower()
        else:
            actions = []
            if "predict" in prompt.lower() or "forecast" in prompt.lower():
                actions.append("forecast")
            if "strategy" in prompt.lower() or "optimize" in prompt.lower():
                actions.append("strategy")
            if "hybrid" in prompt.lower() or "combine" in prompt.lower():
                actions.append("hybrid")
            if "backtest" in prompt.lower():
                actions.append("backtest")
            actions = ", ".join(actions)
        
        st.session_state['prompt'] = prompt
        st.session_state['actions'] = actions
        st.write(f"✅ Detected Actions: {actions}")

# ✅ Forecasting Page (Auto-Detect Model Type + Auto-Debug)
if page == "Forecasting":
    st.header("📈 Fully Agentic Forecasting")
    forecast_period = st.number_input("Forecast Period (Days)", min_value=1, max_value=365, value=30)

    if 'prompt' in st.session_state and 'forecast' in st.session_state.get('actions', ''):
        st.write(f"✅ Interpreted Request: {st.session_state['prompt']}")
        prompt = st.session_state.get('prompt', '')

        try:
            y = data['Target'].values
            models = {}

            # ARIMA Model
            model = ARIMA(y, order=(2, 1, 2)).fit()
            models['ARIMA'] = model.forecast(steps=forecast_period)

            # LSTM + Transformer Model (Auto-Detect Functional API)
            scaler = MinMaxScaler()
            scaled_y = scaler.fit_transform(y.reshape(-1, 1))
            X_train, y_train = [], []

            for i in range(60, len(scaled_y)):
                X_train.append(scaled_y[i - 60:i, 0])
                y_train.append(scaled_y[i, 0])

            X_train = np.array(X_train).reshape((len(X_train), 60, 1))
            y_train = np.array(y_train)

            # ✅ Only use MultiHeadAttention if specified in the prompt
            if prompt and "multiheadattention" in prompt.lower():
                input_layer = Input(shape=(60, 1))
                lstm_out = LSTM(64, return_sequences=True)(input_layer)
                attention_out = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
                lstm_out_2 = LSTM(32)(attention_out)
                output_layer = Dense(1)(lstm_out_2)
                model = Model(inputs=input_layer, outputs=output_layer)
                st.write("✅ Using LSTM + MultiHeadAttention Model")
            else:
                model = Sequential()
                model.add(LSTM(64, return_sequences=True, input_shape=(60, 1)))
                model.add(LSTM(32))
                model.add(Dense(1))
                st.write("✅ Using Standard LSTM Model")

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

            lstm_forecast = model.predict(X_train[-forecast_period:]).flatten()
            models['LSTM+Transformer'] = scaler.inverse_transform(lstm_forecast.reshape(-1, 1)).flatten()

            # XGBoost Model
            X = np.arange(len(y)).reshape(-1, 1)
            model = XGBRegressor()
            model.fit(X, y)
            models['XGBoost'] = model.predict(np.arange(len(y), len(y) + forecast_period).reshape(-1, 1))

            st.write("✅ Forecast Results:")
            for model_name, forecast in models.items():
                st.write(f"{model_name}: {forecast[:5]}...")
        except Exception as e:
            st.error("❌ An error occurred during forecasting.")
            st.text("Error Details:")
            st.text(traceback.format_exc())

# ✅ Auto-Debug Mode (Error Tracking)
st.sidebar.subheader("🚀 Debugging Info")
if st.sidebar.checkbox("Show Debug Log"):
    st.sidebar.text("Auto-Debugging Enabled")
    st.sidebar.text("If any error occurs, it will be shown here.")

st.sidebar.write("🚀 Built with Fully Agentic Automation + Auto-Debugging")
