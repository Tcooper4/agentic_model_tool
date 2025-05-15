
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

def evaluate_model(model, X_test, y_test, task_type="classification"):
    if task_type == "classification":
        predictions = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average='weighted'),
            "recall": recall_score(y_test, predictions, average='weighted'),
            "f1_score": f1_score(y_test, predictions, average='weighted')
        }
    elif task_type == "regression":
        predictions = model.predict(X_test)
        return {
            "mse": mean_squared_error(y_test, predictions)
        }
