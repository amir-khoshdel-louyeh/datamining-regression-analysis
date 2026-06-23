"""
evaluate.py
===========
Evaluation metrics and result visualizations (model comparison chart,
feature importance chart).
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(y_true, y_pred) -> dict:
    """
    Compute MSE and R^2 for a set of predictions.

    Returns
    -------
    dict with keys 'mse' and 'r2'.
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def plot_model_comparison(results: pd.DataFrame, output_path: str) -> None:
    """
    Save a side-by-side bar chart comparing models on MSE and R^2.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain columns 'Model', 'MSE', 'R2'.
    output_path : str
        File path to save the PNG to.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#4C72B0", "#55A868"]

    axes[0].bar(results["Model"], results["MSE"], color=colors)
    axes[0].set_title("Mean Squared Error (lower is better)")
    axes[0].set_ylabel("MSE")

    axes[1].bar(results["Model"], results["R2"], color=colors)
    axes[1].set_title("R-squared (higher is better)")
    axes[1].set_ylabel("R^2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(importances: pd.Series, output_path: str) -> None:
    """
    Save a horizontal bar chart of feature importances.

    Parameters
    ----------
    importances : pd.Series
        Index = feature names, values = importance scores.
    output_path : str
        File path to save the PNG to.
    """
    importances = importances.sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="#55A868")
    plt.title("Feature Importance - Random Forest Regressor")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
