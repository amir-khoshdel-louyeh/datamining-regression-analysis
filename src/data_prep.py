"""
data_prep.py
============
Handles dataset loading and cleaning for the House Price Prediction project.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data() -> pd.DataFrame:
    """
    Load the California Housing dataset from scikit-learn.

    Returns
    -------
    pd.DataFrame
        Full dataset including the target column 'MedHouseVal'.
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean the raw housing dataframe.

    Steps performed:
        1. Report missing values (dataset is clean, but we check anyway).
        2. Drop exact duplicate rows.
        3. Remove rows where the target is capped at its artificial ceiling
           ($500,001 -> stored as 5.00001 in units of $100,000), which is a
           known data-collection artifact in this dataset.
        4. Drop rows with non-physical values (zero/negative rooms,
           bedrooms, or occupancy).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by load_data().
    verbose : bool
        If True, print a summary of cleaning actions.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    if verbose:
        print("Missing values per column:")
        print(df.isnull().sum())

    n_before = len(df)
    df = df.drop_duplicates()
    if verbose:
        print(f"Duplicate rows removed: {n_before - len(df)}")

    capped_value = df["MedHouseVal"].max()
    n_before = len(df)
    df = df[df["MedHouseVal"] < capped_value]
    if verbose:
        print(f"Capped target rows removed: {n_before - len(df)}")

    n_before = len(df)
    df = df[(df["AveRooms"] > 0) & (df["AveOccup"] > 0) & (df["AveBedrms"] > 0)]
    if verbose:
        print(f"Non-physical rows removed: {n_before - len(df)}")
        print(f"Final cleaned dataset shape: {df.shape}")

    return df
