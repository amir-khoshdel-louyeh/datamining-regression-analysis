"""
=============================================================================
Data Mining Mini-Project: House Price Prediction (Regression)
Dataset: California Housing
Models : k-Nearest Neighbors (Baseline) vs Random Forest Regressor
=============================================================================

This is the entry point of the project. It orchestrates the modular
pipeline implemented in src/:

    src/data_prep.py        -> load_data, clean_data
    src/feature_analysis.py -> correlation, heatmap, feature selection
    src/models.py           -> train_knn, train_random_forest
    src/evaluate.py         -> evaluate_model, comparison & importance plots

Run from the project root with:
    python main.py

Author: Amir Khoshdel Louyeh
Course: Data Mining
=============================================================================
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_prep import load_data, clean_data
from src.feature_analysis import (
    compute_correlation,
    plot_correlation_heatmap,
    select_features,
)
from src.models import train_knn, train_random_forest
from src.evaluate import (
    evaluate_model,
    plot_model_comparison,
    plot_feature_importance,
)

RANDOM_STATE = 42
TARGET_COL = "MedHouseVal"
CORR_THRESHOLD = 0.05
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Loading the California Housing dataset")
    print("=" * 70)
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # -----------------------------------------------------------------
    # 2. DATA CLEANING
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Data Cleaning")
    print("=" * 70)
    df = clean_data(df)

    # -----------------------------------------------------------------
    # 3. FEATURE ANALYSIS / CORRELATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Feature Analysis & Correlation")
    print("=" * 70)
    corr_matrix = compute_correlation(df)
    print("Correlation of each feature with target:")
    print(corr_matrix[TARGET_COL].sort_values(ascending=False))

    heatmap_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plot_correlation_heatmap(corr_matrix, heatmap_path)
    print(f"Saved: {heatmap_path}")

    # -----------------------------------------------------------------
    # 4. FEATURE SELECTION
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Feature Selection")
    print("=" * 70)
    selected_features, dropped_features = select_features(
        corr_matrix, TARGET_COL, threshold=CORR_THRESHOLD
    )
    print(f"Correlation threshold: |r| >= {CORR_THRESHOLD}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Dropped features ({len(dropped_features)}): {dropped_features}")

    X = df[selected_features]
    y = df[TARGET_COL]

    # -----------------------------------------------------------------
    # 5. TRAIN/TEST SPLIT AND SCALING
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------------------------------------
    # 6. MODEL 1: KNN (BASELINE)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Training Baseline Model - K-Nearest Neighbors")
    print("=" * 70)
    knn_model, best_k = train_knn(X_train_scaled, y_train)
    print(f"Best k found via 5-fold CV: {best_k}")

    knn_pred = knn_model.predict(X_test_scaled)
    knn_metrics = evaluate_model(y_test, knn_pred)
    print(f"KNN Test MSE: {knn_metrics['mse']:.4f}")
    print(f"KNN Test R^2: {knn_metrics['r2']:.4f}")

    # -----------------------------------------------------------------
    # 7. MODEL 2: RANDOM FOREST
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Training Comparison Model - Random Forest Regressor")
    print("=" * 70)
    rf_model = train_random_forest(X_train_scaled, y_train)

    rf_pred = rf_model.predict(X_test_scaled)
    rf_metrics = evaluate_model(y_test, rf_pred)
    print(f"Random Forest Test MSE: {rf_metrics['mse']:.4f}")
    print(f"Random Forest Test R^2: {rf_metrics['r2']:.4f}")

    # -----------------------------------------------------------------
    # 8. MODEL COMPARISON
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: Model Comparison Summary")
    print("=" * 70)
    results = pd.DataFrame(
        {
            "Model": ["KNN (Baseline)", "Random Forest"],
            "MSE": [knn_metrics["mse"], rf_metrics["mse"]],
            "R2": [knn_metrics["r2"], rf_metrics["r2"]],
        }
    )
    print(results.to_string(index=False))

    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plot_model_comparison(results, comparison_path)
    print(f"Saved: {comparison_path}")

    # -----------------------------------------------------------------
    # 9. FEATURE IMPORTANCE (Random Forest)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8: Feature Importance Visualization (Random Forest)")
    print("=" * 70)
    importances = pd.Series(rf_model.feature_importances_, index=selected_features)
    importance_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plot_feature_importance(importances, importance_path)
    print(f"Saved: {importance_path}")
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False))

    # Also save the numeric results table for the report
    results_path = os.path.join(OUTPUT_DIR, "model_results.csv")
    results.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
