from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler

from .utils import get_logger, load_config

LOGGER = get_logger(__name__)


class OutlierWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> "OutlierWinsorizer":
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        return list(input_features) if input_features is not None else []


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, base_year: int = 1990, scale_factor: float = 30.0):
        self.base_year = base_year
        self.scale_factor = scale_factor

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series | None = None) -> "TemporalFeatureEngineer":
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            values = pd.to_numeric(X.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        else:
            values = np.asarray(X, dtype=float).reshape(-1)
        normalized = (values - self.base_year) / self.scale_factor
        squared = normalized**2
        decade = np.floor(values / 10.0) * 10.0
        decade_encoded = (decade - self.base_year) / 10.0
        return np.column_stack([normalized, squared, decade_encoded])

    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        return ["year_normalized", "year_squared", "year_decade"]


class InteractionFeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> "InteractionFeatureBuilder":
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("InteractionFeatureBuilder expects at least two numeric columns.")
        rainfall = values[:, 0]
        temperature = values[:, 1]
        rainfall_scale = float(np.nanmax(np.abs(rainfall))) if rainfall.size else 0.0
        temperature_scale = float(np.nanmax(np.abs(temperature))) if temperature.size else 0.0
        rainfall_scale = rainfall_scale if rainfall_scale > 0 else 1.0
        temperature_scale = temperature_scale if temperature_scale > 0 else 1.0
        climate_index = (rainfall / rainfall_scale) * (temperature / temperature_scale)
        rain_temp_ratio = rainfall / np.where(np.abs(temperature) < 1e-8, 1.0, temperature)
        return np.column_stack([values, climate_index, rain_temp_ratio])

    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        base_features = list(input_features) if input_features is not None else ["feature_0", "feature_1"]
        return base_features + ["climate_index", "rain_temp_ratio"]


def build_preprocessor(cfg: Optional[dict] = None) -> ColumnTransformer:
    config = cfg or load_config()
    feature_config = config["features"]
    high_cardinality = feature_config["categorical_high_cardinality"]
    low_cardinality = feature_config["categorical_low_cardinality"]
    skewed_numeric = feature_config["numerical_skewed"]
    regular_numeric = feature_config["numerical_normal"]
    temporal_columns = feature_config["temporal"]
    lower_pct = feature_config["outlier_lower_pct"]
    upper_pct = feature_config["outlier_upper_pct"]
    random_state = config["data"]["random_state"]

    skewed_pipeline = Pipeline(
        steps=[
            ("winsorize", OutlierWinsorizer(lower_pct, upper_pct)),
            ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )

    regular_pipeline = Pipeline(
        steps=[
            ("winsorize", OutlierWinsorizer(lower_pct, upper_pct)),
            ("interaction", InteractionFeatureBuilder()),
            ("scale", RobustScaler()),
        ]
    )

    temporal_pipeline = Pipeline(steps=[("temporal", TemporalFeatureEngineer())])

    high_cardinality_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                CatBoostEncoder(
                    sigma=0.05,
                    a=1.0,
                    random_state=random_state,
                    handle_missing="value",
                    handle_unknown="value",
                ),
            )
        ]
    )

    low_cardinality_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("skewed_numeric", skewed_pipeline, skewed_numeric),
            ("regular_numeric", regular_pipeline, regular_numeric),
            ("temporal", temporal_pipeline, temporal_columns),
            ("high_cardinality", high_cardinality_pipeline, high_cardinality),
            ("low_cardinality", low_cardinality_pipeline, low_cardinality),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
        n_jobs=1,
    )
    LOGGER.info("Preprocessor built successfully")
    return preprocessor


def build_full_pipeline(model: BaseEstimator, cfg: Optional[dict] = None) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor(cfg)), ("model", model)])


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names: list[str] = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            column_list = list(columns) if isinstance(columns, (list, tuple)) else [columns]
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names = transformer.get_feature_names_out(column_list)
                    feature_names.extend(list(names))
                    continue
                except Exception:
                    pass
            feature_names.extend([f"{name}__{column}" for column in column_list])
        return feature_names
