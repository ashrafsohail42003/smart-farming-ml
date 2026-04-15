from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import get_logger, resolve_path, validate_dataframe

LOGGER = get_logger(__name__)


def deep_inspect_data(df: pd.DataFrame, max_categories: int = 10) -> dict[str, dict[str, Any]]:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    report: dict[str, dict[str, Any]] = {}
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        summary: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if total_rows > 0:
            summary["null_ratio"] = float(series.isna().mean())
        if pd.api.types.is_numeric_dtype(series) and summary["unique_count"] > 1:
            summary["skew"] = float(series.dropna().skew())
        elif summary["unique_count"] <= max_categories:
            summary["categories"] = [str(value) for value in series.dropna().astype(str).unique().tolist()]
        report[column] = summary
    return report


def clean_data(df: pd.DataFrame, null_threshold: float = 0.6, auto_fix_strings: bool = True) -> pd.DataFrame:
    if not 0.0 <= null_threshold < 1.0:
        raise ValueError("null_threshold must be in the range [0, 1).")
    cleaned = validate_dataframe(df)
    if auto_fix_strings:
        object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
        for column in object_columns:
            normalized = cleaned[column].astype("string").str.strip()
            cleaned[column] = normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA})
    min_non_null = max(1, int(np.ceil((1.0 - null_threshold) * len(cleaned))))
    cleaned = cleaned.dropna(axis=1, thresh=min_non_null)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def load_and_prepare_ml_data(path: str | Path, null_limit: float = 0.5) -> pd.DataFrame:
    data_path = resolve_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    dataframe = pd.read_csv(data_path, low_memory=False)
    if dataframe.empty:
        raise ValueError(f"Input file is empty: {data_path}")
    cleaned = clean_data(dataframe, null_threshold=null_limit)
    LOGGER.info("Loaded dataset from %s with shape %s", data_path, cleaned.shape)
    return cleaned
