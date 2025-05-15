
from xgboost import XGBClassifier

def create_model(config):
    model = XGBClassifier(
        learning_rate=config['hyperparameters']['learning_rate'],
        max_depth=config['hyperparameters']['max_depth'],
        n_estimators=config['hyperparameters']['n_estimators']
    )
    return model
