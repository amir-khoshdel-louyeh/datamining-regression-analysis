# House Price Prediction — Data Mining Mini-Project

Regression mini-project predicting median house value using the
**California Housing dataset**, comparing a **k-Nearest Neighbors (KNN)**
baseline against a **Random Forest Regressor**.

## Project Structure

```
.
├── main.py                  # Entry point — runs the full pipeline
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── src/
│   ├── __init__.py
│   ├── data_prep.py         # Data loading & cleaning
│   ├── feature_analysis.py  # Correlation analysis, heatmap, feature selection
│   ├── models.py            # KNN and Random Forest training
│   └── evaluate.py          # Metrics (MSE, R²) and result plots
└── outputs/                 # Generated on first run — plots & results table
    ├── correlation_heatmap.png
    ├── model_comparison.png
    ├── feature_importance.png
    └── model_results.csv
```

## Setup

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python main.py
```

The first run downloads the California Housing dataset via scikit-learn
(cached locally afterward, so an internet connection is only required once).
All console output (cleaning summary, correlations, selected features,
metrics, feature importances) prints in order, and four files are written
to `outputs/`.

## Pipeline Overview

| Step | Module | Description |
|------|--------|-------------|
| 1. Load data | `src/data_prep.py` | Fetches the California Housing dataset (20,640 instances, 8 features) |
| 2. Clean data | `src/data_prep.py` | Checks for nulls/duplicates, removes target-capped rows, drops non-physical values |
| 3. Feature analysis | `src/feature_analysis.py` | Computes correlation matrix, saves a heatmap |
| 4. Feature selection | `src/feature_analysis.py` | Keeps features with \|correlation\| ≥ 0.05 against the target |
| 5. Train KNN | `src/models.py` | Grid-searches `k` over `[3,5,7,9,11,15]` via 5-fold CV |
| 6. Train Random Forest | `src/models.py` | 200 trees, `min_samples_leaf=2`, fixed random seed |
| 7. Evaluate | `src/evaluate.py` | Computes MSE and R² for both models, saves comparison chart |
| 8. Feature importance | `src/evaluate.py` | Plots Random Forest feature importances |

## Why KNN and Random Forest?

- **KNN (baseline):** simple, makes no distributional assumptions, and is a
  standard sanity-check baseline in regression studies. Its main weakness —
  sensitivity to feature scale and the curse of dimensionality — sets up a
  clear bar for the comparison model to clear.
- **Random Forest:** captures non-linear relationships and feature
  interactions (e.g., income combined with location) without manual feature
  engineering, is robust to outliers, and provides interpretable feature
  importances — useful for the feature-analysis requirement of this
  assignment.

## Evaluation Metrics

- **MSE (Mean Squared Error):** average squared difference between
  predicted and actual house values — penalizes large errors more heavily.
- **R² (Coefficient of Determination):** proportion of variance in house
  prices explained by the model (closer to 1 is better).

## Notes

- Random seed is fixed (`RANDOM_STATE = 42`) throughout for reproducibility.
- `outputs/model_results.csv` contains the final MSE/R² table — use these
  numbers directly in your project report.
