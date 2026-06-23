"""
feature_analysis.py
====================
Correlation analysis, correlation heatmap, and correlation-based
feature selection.
"""

from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Return the full correlation matrix of the dataframe."""
    return df.corr()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str) -> None:
    """
    Save a correlation heatmap to disk.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Output of compute_correlation().
    output_path : str
        File path (including filename) to save the PNG to.
    """
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap - California Housing Dataset")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def select_features(
    corr_matrix: pd.DataFrame,
    target_col: str,
    threshold: float = 0.05,
) -> Tuple[List[str], List[str]]:
    """
    Select features whose absolute correlation with the target is
    at or above a given threshold.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Output of compute_correlation().
    target_col : str
        Name of the target column.
    threshold : float
        Minimum absolute correlation required to keep a feature.

    Returns
    -------
    (selected, dropped) : Tuple[List[str], List[str]]
        Lists of feature names that passed/failed the threshold.
    """
    target_corr = corr_matrix[target_col].drop(target_col).abs()
    selected = target_corr[target_corr >= threshold].index.tolist()
    dropped = target_corr[target_corr < threshold].index.tolist()
    return selected, dropped
