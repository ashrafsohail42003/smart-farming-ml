from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOGGER_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_directory(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str, log_dir: str | Path = "logs") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_directory = ensure_directory(log_dir)
    log_path = log_directory / "pipeline.log"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(LOGGER_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOGGER_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


LOGGER = get_logger(__name__)


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config structure in {path}")
    return config


def validate_dataframe(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    cleaned = df.copy()
    cleaned.columns = [str(column).strip().lower().replace(" ", "_") for column in cleaned.columns]
    unnamed_columns = [column for column in cleaned.columns if "unnamed" in column]
    if unnamed_columns:
        cleaned = cleaned.drop(columns=unnamed_columns)
    if target_col:
        normalized_target = target_col.strip().lower().replace(" ", "_")
        if normalized_target not in cleaned.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    return cleaned


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    validated = validate_dataframe(df, target_col)
    target_name = target_col.strip().lower().replace(" ", "_")
    features = validated.drop(columns=[target_name]).copy()
    target = pd.to_numeric(validated[target_name], errors="coerce")
    valid_mask = target.notna()
    features = features.loc[valid_mask].reset_index(drop=True)
    target = target.loc[valid_mask].reset_index(drop=True)
    if features.empty or target.empty:
        raise ValueError("No valid rows remain after cleaning the target column.")
    return features, target


def compute_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, prefix: str = "") -> dict[str, float]:
    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    if true_values.size == 0 or predicted_values.size == 0:
        raise ValueError("Empty arrays are not valid for metric computation.")
    if true_values.shape[0] != predicted_values.shape[0]:
        raise ValueError("Target and prediction arrays must have the same length.")
    rmse = float(np.sqrt(mean_squared_error(true_values, predicted_values)))
    mae = float(mean_absolute_error(true_values, predicted_values))
    r2 = float(r2_score(true_values, predicted_values))
    non_zero_mask = true_values != 0
    mape = float(np.mean(np.abs((true_values[non_zero_mask] - predicted_values[non_zero_mask]) / true_values[non_zero_mask])) * 100.0) if np.any(non_zero_mask) else 0.0
    clipped_true = np.maximum(true_values, 0.0)
    clipped_pred = np.maximum(predicted_values, 0.0)
    rmsle = float(np.sqrt(mean_squared_error(np.log1p(clipped_true), np.log1p(clipped_pred))))
    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}mape": mape,
        f"{prefix}r2": r2,
        f"{prefix}rmsle": rmsle,
    }


def print_metrics(metrics: dict[str, float], title: str = "Evaluation Results") -> None:
    LOGGER.info("%s | %s", title, metrics)


def _prepare_figure(save_path: str | Path | None) -> Path | None:
    if save_path is None:
        return None
    path = resolve_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_predictions_vs_actual(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    title: str = "Predicted vs Actual",
    save_path: str | Path | None = None,
) -> None:
    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    if true_values.size == 0:
        return
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(true_values, predicted_values, alpha=0.5, s=20, color="steelblue")
    limits = [float(min(true_values.min(), predicted_values.min())), float(max(true_values.max(), predicted_values.max()))]
    axis.plot(limits, limits, linestyle="--", linewidth=1.5, color="darkred")
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")
    axis.set_title(title)
    output_path = _prepare_figure(save_path)
    if output_path is not None:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    save_path: str | Path | None = None,
) -> None:
    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    if true_values.size == 0:
        return
    residuals = true_values - predicted_values
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(predicted_values, residuals, alpha=0.5, s=20, color="coral")
    axes[0].axhline(0.0, color="black", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    sns.histplot(residuals, kde=True, ax=axes[1], color="teal")
    axes[1].set_xlabel("Residual")
    output_path = _prepare_figure(save_path)
    if output_path is not None:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: str | Path | None = None,
) -> None:
    importance_values = np.asarray(importances, dtype=float).reshape(-1)
    if importance_values.size == 0:
        return
    top_n = max(1, min(top_n, importance_values.size, len(feature_names)))
    indices = np.argsort(importance_values)[-top_n:]
    labels = [feature_names[index] for index in indices]
    values = importance_values[indices]
    figure, axis = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    axis.barh(labels, values, color=sns.color_palette("viridis", top_n))
    axis.set_title(title)
    axis.set_xlabel("Importance")
    output_path = _prepare_figure(save_path)
    if output_path is not None:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_cv_scores(scores: np.ndarray, metric_name: str = "RMSE", save_path: str | Path | None = None) -> None:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        return
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(range(1, values.size + 1), values, edgecolor="black")
    axis.axhline(values.mean(), linestyle="--", linewidth=1.5, color="darkred")
    axis.set_xlabel("Fold")
    axis.set_ylabel(metric_name)
    axis.set_title(f"Cross-Validation {metric_name}")
    output_path = _prepare_figure(save_path)
    if output_path is not None:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_artifact(obj: object, path: str | Path) -> Path:
    artifact_path = resolve_path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, artifact_path)
    return artifact_path


def load_artifact(path: str | Path) -> object:
    artifact_path = resolve_path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    return joblib.load(artifact_path)


def check_data_drift(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.1) -> dict[str, float]:
    if reference.empty or current.empty:
        raise ValueError("Reference and current dataframes must both be non-empty.")
    report: dict[str, float] = {}
    numeric_columns = reference.select_dtypes(include=np.number).columns.intersection(current.columns)
    for column in numeric_columns:
        reference_mean = float(reference[column].mean())
        current_mean = float(current[column].mean())
        drift = abs(current_mean - reference_mean) / (abs(reference_mean) + 1e-8)
        report[column] = drift
        if drift > threshold:
            LOGGER.warning("Detected data drift for %s: %.4f", column, drift)
    return report
