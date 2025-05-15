
import optuna

def optimize_model(model, X_train, y_train):
    def objective(trial):
        learning_rate = trial.suggest_loguniform("learning_rate", 0.0001, 0.1)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        model.set_params(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    return model
