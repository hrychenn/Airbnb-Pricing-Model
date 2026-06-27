"""
Train, evaluate, and serialize the pricing models.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred_log: np.ndarray) -> dict:
    y_pred = np.expm1(y_pred_log)
    return {
        "mape": mape(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def naive_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
    return evaluate(y_test, np.log1p(pred))


def save_model(pipeline, path: str) -> None:
    joblib.dump(pipeline, path)
    print(f"Model saved → {path}")


def load_model(path: str):
    return joblib.load(path)
