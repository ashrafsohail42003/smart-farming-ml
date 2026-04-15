import numpy as np
import pandas as pd
import pytest

from src.features import OutlierWinsorizer, TemporalFeatureEngineer, build_full_pipeline, build_preprocessor
from src.utils import load_config


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    row_count = 200
    return pd.DataFrame(
        {
            "area": rng.choice(["Albania", "Algeria", "Germany", "Egypt"], row_count),
            "item": rng.choice(["Maize", "Wheat", "Rice"], row_count),
            "year": rng.integers(1990, 2015, row_count),
            "average_rain_fall_mm_per_year": rng.uniform(200, 2000, row_count),
            "pesticides_tonnes": rng.exponential(100, row_count),
            "avg_temp": rng.uniform(5, 35, row_count),
            "hg/ha_yield": rng.uniform(1000, 80000, row_count),
        }
    )


@pytest.fixture
def cfg():
    return load_config()


class TestOutlierWinsorizer:
    def test_fit_transform(self):
        values = np.array([[1], [2], [100], [3], [200]], dtype=float)
        transformer = OutlierWinsorizer(lower=0.1, upper=0.9)
        transformed = transformer.fit_transform(values)
        assert transformed.shape == values.shape
        assert transformed.max() <= 160.0
        assert transformed.min() >= 1.4

    def test_no_leakage(self):
        train_values = np.array([[10], [20], [30], [40], [50]], dtype=float)
        test_values = np.array([[1000]], dtype=float)
        transformer = OutlierWinsorizer(0.01, 0.99)
        transformer.fit(train_values)
        transformed = transformer.transform(test_values)
        assert transformed[0, 0] <= 50.0

    def test_handles_1d(self):
        values = np.array([1, 2, 3, 100], dtype=float)
        transformer = OutlierWinsorizer()
        transformed = transformer.fit_transform(values)
        assert transformed.shape == (4, 1)


class TestTemporalFeatureEngineer:
    def test_output_shape(self):
        values = pd.DataFrame({"year": [1990, 2000, 2010, 2020]})
        transformer = TemporalFeatureEngineer()
        transformed = transformer.fit_transform(values)
        assert transformed.shape == (4, 3)

    def test_feature_names(self):
        transformer = TemporalFeatureEngineer()
        names = transformer.get_feature_names_out()
        assert names == ["year_normalized", "year_squared", "year_decade"]

    def test_normalization_range(self):
        values = pd.DataFrame({"year": [1990, 2020]})
        transformer = TemporalFeatureEngineer()
        transformed = transformer.transform(values)
        assert transformed[0, 0] == pytest.approx(0.0)
        assert transformed[1, 0] == pytest.approx(1.0)


class TestPreprocessor:
    def test_build_preprocessor(self, cfg):
        assert build_preprocessor(cfg) is not None

    def test_fit_transform(self, sample_df, cfg):
        features = sample_df.drop("hg/ha_yield", axis=1)
        target = sample_df["hg/ha_yield"]
        preprocessor = build_preprocessor(cfg)
        transformed = preprocessor.fit_transform(features, target)
        assert transformed.shape[0] == len(features)
        assert not np.isnan(transformed).any()

    def test_no_train_test_leakage(self, sample_df, cfg):
        features = sample_df.drop("hg/ha_yield", axis=1)
        target = sample_df["hg/ha_yield"]
        train_features = features.iloc[:150]
        test_features = features.iloc[150:]
        train_target = target.iloc[:150]
        preprocessor = build_preprocessor(cfg)
        preprocessor.fit(train_features, train_target)
        transformed = preprocessor.transform(test_features)
        assert transformed.shape[0] == len(test_features)

    def test_handles_unseen_categories(self, sample_df, cfg):
        features = sample_df.drop("hg/ha_yield", axis=1).copy()
        target = sample_df["hg/ha_yield"]
        preprocessor = build_preprocessor(cfg)
        preprocessor.fit(features, target)
        features.loc[features.index[0], "area"] = "NEW_UNSEEN_COUNTRY"
        transformed = preprocessor.transform(features)
        assert transformed.shape[0] == len(features)


class TestFullPipeline:
    def test_fit_predict(self, sample_df, cfg):
        from lightgbm import LGBMRegressor

        features = sample_df.drop("hg/ha_yield", axis=1)
        target = sample_df["hg/ha_yield"]
        model = LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        pipeline = build_full_pipeline(model, cfg)
        pipeline.fit(features, target)
        predictions = pipeline.predict(features)
        assert predictions.shape == (len(features),)
        assert not np.isnan(predictions).any()

    def test_predictions_non_negative(self, sample_df, cfg):
        from lightgbm import LGBMRegressor

        features = sample_df.drop("hg/ha_yield", axis=1)
        target = sample_df["hg/ha_yield"]
        model = LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        pipeline = build_full_pipeline(model, cfg)
        pipeline.fit(features, target)
        predictions = np.maximum(pipeline.predict(features), 0.0)
        assert (predictions >= 0.0).all()
