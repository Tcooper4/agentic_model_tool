
from xgboost import XGBClassifier

def create_model(config):
    model = XGBClassifier(
        learning_rate=config['hyperparameters']['learning_rate'],
        max_depth=config['hyperparameters'].get('max_depth', 6),
        n_estimators=config['hyperparameters'].get('n_estimators', 100)
    )
    return model
