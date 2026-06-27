"""
Optuna hyperparameter search for XGBoost.
"""
import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def xgb_objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    return float(-scores.mean())


def run_study(X, y, n_trials: int = 50) -> optuna.Study:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: xgb_objective(trial, X, y), n_trials=n_trials)
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study
