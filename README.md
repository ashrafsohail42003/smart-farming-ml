# Smart Farming ML

![Python](https://img.shields.io/badge/Python-3.14-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)
![MLflow](https://img.shields.io/badge/MLflow-3.11.1-0194E2)
![Status](https://img.shields.io/badge/Status-Working-success)

Production-oriented crop yield prediction pipeline with configurable training modes, leakage-aware preprocessing, model comparison, Optuna tuning, MLflow experiment tracking, and automated evaluation artifacts.

## Overview

This repository implements an end-to-end regression pipeline for crop yield prediction using structured agricultural data. The project is built as a modular ML system rather than a notebook workflow: configuration is centralized in `config.yaml`, the training orchestration lives in `main.py`, feature engineering is encapsulated in reusable scikit-learn transformers, model training and evaluation are handled by a dedicated trainer class, and artifacts are persisted for reproducible reuse.

Technically, the project stands out because it combines:

- a CLI-driven orchestration layer
- reusable preprocessing inside a `Pipeline`
- train/test separation with cross-validation
- a baseline model for sanity checks
- Optuna-based hyperparameter optimization
- automated metric computation and plots
- MLflow-based experiment tracking
- unit tests for preprocessing and pipeline behavior

## Key Features

- Multiple execution modes through a single CLI entry point: `full`, `train`, `tune`, `compare`, `stack`, and `evaluate`
- Four supported regressors: `LightGBM`, `XGBoost`, `CatBoost`, and `Ridge`
- Baseline benchmarking with `DummyRegressor`
- Leakage-aware preprocessing built with `ColumnTransformer`
- Outlier winsorization using learned train-set quantile bounds
- Temporal feature engineering from `year`
- Numeric interaction features between rainfall and temperature
- Mixed categorical encoding strategy:
  - `CatBoostEncoder` for high-cardinality categorical features
  - `OneHotEncoder` for low-cardinality categorical features
- Cross-validation with fold-level RMSE tracking
- Optuna hyperparameter tuning for tree-based models
- Evaluation outputs including:
  - RMSE
  - MAE
  - MAPE
  - R2
  - RMSLE
- Auto-generated visual reports:
  - CV score plots
  - predicted vs actual plots
  - residual diagnostics
  - feature importance plots when available
- MLflow tracking to a local SQLite backend
- Artifact persistence for trained pipelines

## Architecture / Pipeline

The runtime flow is orchestrated from `main.py` and centered around `ModelTrainer`.

### 1. Configuration loading

The application loads configuration from `config.yaml`, which defines:

- data paths
- target column
- train/test split ratio
- feature groups
- model hyperparameters
- Optuna settings
- MLflow settings
- output directories

### 2. Data loading

`ModelTrainer.load_data()` reads the processed dataset from:

```text
data/processed/cleaned_data.csv
```

The data is then split into:

- `X_train`, `X_test`
- `y_train`, `y_test`

using `train_test_split` with:

- `test_size: 0.15`
- `random_state: 42`

### 3. Preprocessing

The preprocessing stack is built in `src/features.py` using a `ColumnTransformer` with explicit column groups from config.

#### Skewed numeric features

Configured for:

- `pesticides_tonnes`

Pipeline:

1. `OutlierWinsorizer`
2. `PowerTransformer(method="yeo-johnson")`

#### Standard numeric features

Configured for:

- `average_rain_fall_mm_per_year`
- `avg_temp`

Pipeline:

1. `OutlierWinsorizer`
2. `InteractionFeatureBuilder`
3. `RobustScaler`

The interaction builder adds:

- `climate_index`
- `rain_temp_ratio`

#### Temporal features

Configured for:

- `year`

`TemporalFeatureEngineer` derives:

- `year_normalized`
- `year_squared`
- `year_decade`

#### High-cardinality categorical features

Configured for:

- `area`

Encoding:

- `CatBoostEncoder`

#### Low-cardinality categorical features

Configured for:

- `item`

Encoding:

- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

### 4. Model training

The full modeling object is always built as:

```text
preprocessor -> model
```

inside a single scikit-learn `Pipeline`. This design ensures that preprocessing is learned on training data and consistently applied during cross-validation, final training, and inference.

### 5. Evaluation

After training, the pipeline is evaluated on the hold-out test split. Predictions are clipped to non-negative values before metric calculation to avoid negative crop yield outputs.

Computed metrics:

- `RMSE`
- `MAE`
- `MAPE`
- `R2`
- `RMSLE`

Generated plots:

- predicted vs actual
- residual analysis
- feature importance when the estimator exposes `feature_importances_`

## Models

The project supports the following models through `get_model()`:

### LightGBM

Default primary model. It is also the model used in the executed training runs and result logs currently present in the repository.

Why it fits this project:

- strong performance on tabular data
- fast training
- handles non-linear interactions well
- supports robust regularization and tuning

### XGBoost

Implemented as an alternative tree boosting model with histogram-based training.

Why it is included:

- strong tabular baseline/competitor
- good regularization controls
- useful for comparison against LightGBM

### CatBoost

Included as a third gradient boosting option.

Why it is included:

- strong performance on heterogeneous structured data
- useful alternative in comparative experiments

### Ridge

Included as a lightweight linear baseline model.

Why it is included:

- cheap and interpretable benchmark
- useful to compare linear vs boosted-tree behavior

### Stacking Ensemble

The `stack` mode trains a `StackingRegressor` using:

- LightGBM
- CatBoost
- XGBoost

with:

- `Ridge(alpha=1.0)` as the final estimator

## Results

The repository has already been executed successfully on Windows, and the following metrics were produced from actual runs.

### Train mode (`python main.py --mode train --model lgbm`)

- `RMSE = 14053.74`
- `MAE = 7021.33`
- `R2 = 0.97237`

### Full mode (`python main.py --mode full --model lgbm`)

- `Baseline RMSE = 84561.47`
- `CV RMSE mean = 12819.86 ± 485.25`
- `Test RMSE = 12402.32`
- `MAE = 5902.77`
- `MAPE = 15.34%`
- `R2 = 0.97848`

Interpretation:

- the tuned model dramatically outperforms the dummy baseline
- cross-validation and test performance are closely aligned
- the gap between CV and test is small, which is a good sign for generalization stability

Artifacts produced from actual execution include:

- `models/lgbm_final_pipeline.pkl`
- `reports/lgbm_optuna_trials.csv`
- `reports/figures/lgbm_cv_scores.png`
- `reports/figures/lgbm_pred_vs_actual.png`
- `reports/figures/lgbm_residuals.png`
- `reports/figures/lgbm_feature_importance.png`
- `logs/pipeline.log`
- `mlflow.db`

## Project Structure

```text
smart-farming-ml/
├── config.yaml
├── main.py
├── requirements.txt
├── README.md
├── mlflow.db
├── data/
│   ├── raw/
│   └── processed/
│       └── cleaned_data.csv
├── logs/
│   └── pipeline.log
├── models/
│   └── lgbm_final_pipeline.pkl
├── reports/
│   ├── lgbm_optuna_trials.csv
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── utils.py
└── tests/
    └── test_features.py
```

### Module responsibilities

- `main.py`: CLI entry point and mode orchestration
- `src/data.py`: dataset validation, cleaning, and inspection helpers
- `src/features.py`: custom transformers and preprocessing pipeline construction
- `src/model.py`: training, tuning, comparison, stacking, evaluation, artifact logging
- `src/utils.py`: configuration loading, metrics, plotting, logging, artifact persistence
- `tests/test_features.py`: preprocessing and pipeline unit tests

## Installation

### Windows PowerShell

```powershell
cd C:\Users\MSI\Desktop\project\smart-farming-ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Usage (CLI)

All runtime modes are exposed through `main.py`.

### Train

Runs cross-validation, trains the final model with config parameters, and evaluates on the hold-out test split.

```powershell
python .\main.py --mode train --model lgbm
python .\main.py --mode train --model catboost
python .\main.py --mode train --model xgboost
python .\main.py --mode train --model ridge
```

### Full

Runs the complete workflow:

1. baseline model
2. cross-validation
3. Optuna tuning
4. final training with best parameters
5. test evaluation

```powershell
python .\main.py --mode full --model lgbm
```

### Tune

Tunes a model with Optuna, retrains using the best parameters, and evaluates it.

```powershell
python .\main.py --mode tune --model lgbm --trials 20
python .\main.py --mode tune --model xgboost --trials 20
python .\main.py --mode tune --model catboost --trials 20
```

### Compare

Trains and evaluates all implemented tree models and exports a comparison table.

```powershell
python .\main.py --mode compare
```

### Stack

Builds and evaluates the stacking ensemble.

```powershell
python .\main.py --mode stack
```

### Evaluate

Loads a saved pipeline from `models/` and evaluates it on the configured test split.

```powershell
python .\main.py --mode evaluate --model lgbm
```

### Optional arguments

Override the config path:

```powershell
python .\main.py --mode train --model lgbm --config .\config.yaml
```

Override the processed dataset path:

```powershell
python .\main.py --mode train --model lgbm --data .\data\processed\cleaned_data.csv
```

## Experiments & Tracking

The project integrates MLflow with the following configuration:

- tracking URI: `sqlite:///mlflow.db`
- experiment name: `crop-yield-prediction`
- registry name: `CropYieldPredictor`

MLflow is used to log:

- baseline metrics
- CV summaries
- Optuna tuning results
- evaluation metrics
- generated plots
- model artifacts when logging succeeds

Tracking is designed to be resilient: MLflow setup and logging failures are caught and logged as warnings rather than crashing the main training flow.

## Testing

The test suite currently focuses on feature engineering and pipeline correctness.

Run all tests:

```powershell
python -m pytest .\tests -q -p no:cacheprovider
```

### What is validated

- `OutlierWinsorizer` clips values correctly
- winsorization is learned from train data and reused on test data
- 1D numeric inputs are handled safely
- temporal feature engineering returns the correct shape and expected feature names
- year normalization behaves as expected
- the preprocessor builds successfully from config
- transformed output does not contain `NaN` values
- preprocessing can be fit on train and applied to test without leakage issues
- unseen categories are handled during transform
- the full model pipeline fits and predicts successfully
- final predictions can be constrained to non-negative values

## Future Improvements

The current repository is a strong ML pipeline project, but the following extensions would make it closer to a production deployment package:

- add a dedicated inference API or batch prediction script
- add CI workflows for tests and linting
- add Docker support for reproducible deployment
- add a richer README section on dataset provenance and business context
- expand test coverage to training orchestration and artifact generation
- reduce remaining runtime warnings from downstream libraries
- add data and model versioning
- expose comparison and evaluation summaries in a richer report format

## Author

**Ashraf Sohail Al-Kahlout (أشرف سهيل الكحلوت)**


---

If you are reviewing this repository as a recruiter or engineer: this project demonstrates practical ML engineering skills beyond notebook experimentation, including modular pipeline design, reusable preprocessing, experiment tracking, artifact persistence, CLI orchestration, and validation through tests.
