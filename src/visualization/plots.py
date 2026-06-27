"""Reusable plotting helpers."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_price_distribution(prices: np.ndarray, ax=None) -> plt.Axes:
    ax = ax or plt.gca()
    sns.histplot(prices, bins=80, log_scale=(False, True), ax=ax)
    ax.set_xlabel("Nightly Price ($)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Price Distribution")
    return ax


def plot_neighbourhood_prices(df, top_n: int = 20, ax=None) -> plt.Axes:
    ax = ax or plt.gca()
    top = (
        df.groupby("neighbourhood_cleansed")["price"]
        .median()
        .nlargest(top_n)
        .reset_index()
    )
    sns.barplot(data=top, x="price", y="neighbourhood_cleansed", ax=ax)
    ax.set_title(f"Top {top_n} Neighbourhoods by Median Price")
    ax.set_xlabel("Median Nightly Price ($)")
    ax.set_ylabel("")
    return ax


def plot_actual_vs_predicted(y_true, y_pred, ax=None) -> plt.Axes:
    ax = ax or plt.gca()
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Actual vs. Predicted")
    return ax
