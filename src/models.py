"""
models.py
=========
Model training utilities for the baseline (KNN) and comparison
(Random Forest) regressors.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42


def train_knn(X_train, y_train, k_values=(3, 5, 7, 9, 11, 15), cv: int = 5):
    """
    Train a KNN regressor, selecting the best 'k' via cross-validated
    grid search on negative MSE.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features (should already be scaled) and target.
    k_values : iterable of int
        Candidate values of n_neighbors to try.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    best_model : KNeighborsRegressor
        The fitted estimator with the best-found k.
    best_k : int
        The selected value of n_neighbors.
    """
    param_grid = {"n_neighbors": list(k_values)}
    grid = GridSearchCV(
        KNeighborsRegressor(),
        param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_["n_neighbors"]


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 200,
    max_depth=None,
    min_samples_leaf: int = 2,
):
    """
    Train a Random Forest regressor with sensible default hyperparameters.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and target.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum tree depth (None = unlimited).
    min_samples_leaf : int
        Minimum samples required at a leaf node (helps avoid overfitting).

    Returns
    -------
    RandomForestRegressor
        The fitted estimator.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
