
import optuna
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def create_model(config):
    model_type = config.get("model_type", "LogisticRegression")
    
    if model_type == "LogisticRegression":
        return LogisticRegression()
    elif model_type == "RandomForest":
        return RandomForestClassifier(n_estimators=config.get("n_estimators", 100))
    elif model_type == "XGBoost":
        return XGBClassifier()
    elif model_type == "LLM":
        model_name = config.get("model_name", "distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model, tokenizer
    else:
        raise ValueError("Unsupported model type: " + model_type)
