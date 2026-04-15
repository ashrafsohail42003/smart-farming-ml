from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from mlflow.models.signature import infer_signature
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .features import build_full_pipeline, get_feature_names
from .utils import (
    compute_metrics,
    get_logger,
    plot_cv_scores,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_residuals,
    print_metrics,
    resolve_path,
    save_artifact,
    split_features_target,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
LOGGER = get_logger(__name__)


def get_model(model_name: str, cfg: dict, overrides: Optional[dict] = None):
    model_config = dict(cfg["models"].get(model_name, {}))
    if overrides:
        model_config.update(overrides)
    if model_name == "lgbm":
        return LGBMRegressor(**model_config)
    if model_name == "catboost":
        return CatBoostRegressor(**model_config)
    if model_name == "xgboost":
        return xgb.XGBRegressor(**model_config)
    if model_name == "ridge":
        return Ridge(alpha=float(model_config.get("alpha", 1.0)))
    raise ValueError(f"Unsupported model: {model_name}")


class ModelTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.target_col = cfg["features"]["target"]
        self.random_state = int(cfg["data"]["random_state"])
        self.test_size = float(cfg["data"]["test_size"])
        self.paths = cfg["paths"]
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.best_pipeline: Optional[Pipeline] = None
        self.best_params: Optional[dict] = None
        self.feature_names: list[str] = []
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        try:
            mlflow.set_tracking_uri(self.cfg["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.cfg["mlflow"]["experiment_name"])
        except Exception as error:
            LOGGER.warning("MLflow setup failed: %s", error)

    def _start_mlflow_run(self, run_name: str):
        try:
            return mlflow.start_run(run_name=run_name)
        except Exception as error:
            LOGGER.warning("MLflow run start failed: %s", error)
            return None

    def _log_mlflow_artifact(self, path: str | Path) -> None:
        try:
            mlflow.log_artifact(str(resolve_path(path)))
        except Exception as error:
            LOGGER.warning("MLflow artifact logging failed: %s", error)

    def _assert_data_loaded(self) -> None:
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

    def load_data(self, path: str | Path | None = None) -> None:
        data_path = resolve_path(path or self.cfg["data"]["processed_path"])
        dataframe = pd.read_csv(data_path, low_memory=False)
        features, target = split_features_target(dataframe, self.target_col)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
        LOGGER.info("Loaded training data. Train shape=%s, Test shape=%s", self.X_train.shape, self.X_test.shape)

    def train_baseline(self) -> dict[str, float]:
        self._assert_data_loaded()
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(self.X_train, self.y_train)
        predictions = baseline.predict(self.X_test)
        metrics = compute_metrics(self.y_test, predictions, prefix="baseline_")
        print_metrics(metrics, "Baseline")
        run = self._start_mlflow_run("baseline_dummy")
        if run is not None:
            with run:
                mlflow.log_param("model_type", "DummyRegressor")
                mlflow.log_metrics(metrics)
        return metrics

    def train_with_cv(self, model_name: str = "lgbm", log_to_mlflow: bool = True) -> dict[str, float]:
        self._assert_data_loaded()
        cv_folds = int(self.cfg["models"]["cv_folds"])
        model = get_model(model_name, self.cfg)
        pipeline = build_full_pipeline(model, self.cfg)
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_results = cross_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv=splitter,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            return_train_score=True,
            n_jobs=1,
        )
        test_rmse = -cv_results["test_rmse"]
        test_mae = -cv_results["test_mae"]
        test_r2 = cv_results["test_r2"]
        train_rmse = -cv_results["train_rmse"]
        summary = {
            "cv_rmse_mean": float(test_rmse.mean()),
            "cv_rmse_std": float(test_rmse.std()),
            "cv_mae_mean": float(test_mae.mean()),
            "cv_r2_mean": float(test_r2.mean()),
            "cv_r2_std": float(test_r2.std()),
            "overfit_gap": float(train_rmse.mean() - test_rmse.mean()),
        }
        figure_path = Path(self.paths["figures_dir"]) / f"{model_name}_cv_scores.png"
        plot_cv_scores(test_rmse, metric_name="RMSE", save_path=figure_path)
        if log_to_mlflow:
            run = self._start_mlflow_run(f"{model_name}_cv")
            if run is not None:
                with run:
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("cv_folds", cv_folds)
                    mlflow.log_metrics(summary)
                    self._log_mlflow_artifact(figure_path)
        return summary

    def tune_with_optuna(
        self,
        model_name: str = "lgbm",
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        self._assert_data_loaded()
        trial_count = int(n_trials or self.cfg["optuna"]["n_trials"])
        trial_timeout = int(timeout or self.cfg["optuna"]["timeout"])
        cv_folds = int(self.cfg["optuna"]["cv_folds"])
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        def objective(trial: optuna.Trial) -> float:
            if model_name == "lgbm":
                overrides = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                }
            elif model_name == "xgboost":
                overrides = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                }
            elif model_name == "catboost":
                overrides = {
                    "iterations": trial.suggest_int("iterations", 200, 1200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                }
            else:
                raise ValueError(f"Unsupported model for Optuna tuning: {model_name}")
            model = get_model(model_name, self.cfg, overrides=overrides)
            pipeline = build_full_pipeline(model, self.cfg)
            scores = cross_validate(
                pipeline,
                self.X_train,
                self.y_train,
                cv=splitter,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            )
            return float(-scores["test_score"].mean())

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=trial_count, timeout=trial_timeout, show_progress_bar=False)
        self.best_params = dict(study.best_params)
        trials_path = resolve_path(Path(self.paths["reports_dir"]) / f"{model_name}_optuna_trials.csv")
        trials_path.parent.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(trials_path, index=False)
        run = self._start_mlflow_run(f"{model_name}_optuna")
        if run is not None:
            with run:
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("n_trials", trial_count)
                mlflow.log_params(self.best_params)
                mlflow.log_metric("best_cv_rmse", float(study.best_value))
                self._log_mlflow_artifact(trials_path)
        return self.best_params

    def train_final(self, model_name: str = "lgbm", use_best_params: bool = True) -> Pipeline:
        self._assert_data_loaded()
        overrides = self.best_params if use_best_params and self.best_params else None
        model = get_model(model_name, self.cfg, overrides=overrides)
        pipeline = build_full_pipeline(model, self.cfg)
        pipeline.fit(self.X_train, self.y_train)
        self.best_pipeline = pipeline
        try:
            self.feature_names = get_feature_names(pipeline.named_steps["preprocessor"])
        except Exception:
            self.feature_names = []
        save_artifact(pipeline, Path(self.paths["models_dir"]) / f"{model_name}_final_pipeline.pkl")
        return pipeline

    def train_stacking(self) -> Pipeline:
        self._assert_data_loaded()
        estimators = [
            ("lgbm", get_model("lgbm", self.cfg)),
            ("catboost", get_model("catboost", self.cfg)),
            ("xgboost", get_model("xgboost", self.cfg)),
        ]
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            passthrough=False,
            n_jobs=1,
        )
        pipeline = build_full_pipeline(stacking, self.cfg)
        pipeline.fit(self.X_train, self.y_train)
        self.best_pipeline = pipeline
        save_artifact(pipeline, Path(self.paths["models_dir"]) / "stacking_final_pipeline.pkl")
        return pipeline

    def evaluate(
        self,
        pipeline: Optional[Pipeline] = None,
        model_name: str = "final_model",
        log_to_mlflow: bool = True,
    ) -> dict[str, float]:
        self._assert_data_loaded()
        fitted_pipeline = pipeline or self.best_pipeline
        if fitted_pipeline is None:
            raise RuntimeError("No trained pipeline available for evaluation.")
        predictions = np.asarray(fitted_pipeline.predict(self.X_test), dtype=float).reshape(-1)
        predictions = np.maximum(predictions, 0.0)
        metrics = compute_metrics(self.y_test, predictions)
        print_metrics(metrics, f"Test metrics for {model_name}")
        figures_dir = Path(self.paths["figures_dir"])
        prediction_plot = figures_dir / f"{model_name}_pred_vs_actual.png"
        residual_plot = figures_dir / f"{model_name}_residuals.png"
        plot_predictions_vs_actual(self.y_test, predictions, title=f"Predicted vs Actual - {model_name}", save_path=prediction_plot)
        plot_residuals(self.y_test, predictions, save_path=residual_plot)
        self._plot_importance_if_available(fitted_pipeline, model_name, figures_dir)
        if log_to_mlflow:
            run = self._start_mlflow_run(f"{model_name}_evaluation")
            if run is not None:
                with run:
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("test_size", self.test_size)
                    mlflow.log_metrics(metrics)
                    self._log_mlflow_artifact(prediction_plot)
                    self._log_mlflow_artifact(residual_plot)
                    try:
                        signature = infer_signature(self.X_test, predictions)
                        mlflow.sklearn.log_model(fitted_pipeline, artifact_path="model", signature=signature)
                    except Exception as error:
                        LOGGER.warning("MLflow model logging failed: %s", error)
        return metrics

    def _plot_importance_if_available(self, pipeline: Pipeline, model_name: str, figures_dir: Path) -> None:
        try:
            model = pipeline.named_steps["model"]
            candidate = model.estimators_[0] if hasattr(model, "estimators_") and model.estimators_ else model
            if hasattr(candidate, "feature_importances_"):
                importances = np.asarray(candidate.feature_importances_, dtype=float)
                names = self.feature_names or [f"feature_{index}" for index in range(importances.size)]
                plot_feature_importance(
                    feature_names=names[: importances.size],
                    importances=importances,
                    top_n=min(20, importances.size),
                    title=f"Feature Importance - {model_name}",
                    save_path=figures_dir / f"{model_name}_feature_importance.png",
                )
        except Exception as error:
            LOGGER.warning("Feature importance plot skipped: %s", error)

    def compare_models(self) -> pd.DataFrame:
        self._assert_data_loaded()
        results: list[dict[str, float | str]] = []
        for model_name in ["lgbm", "catboost", "xgboost"]:
            try:
                cv_summary = self.train_with_cv(model_name)
                pipeline = self.train_final(model_name, use_best_params=False)
                metrics = self.evaluate(pipeline, model_name)
                results.append(
                    {
                        "model": model_name,
                        "cv_rmse": cv_summary["cv_rmse_mean"],
                        "cv_rmse_std": cv_summary["cv_rmse_std"],
                        "test_rmse": metrics["rmse"],
                        "test_mae": metrics["mae"],
                        "test_r2": metrics["r2"],
                        "test_mape": metrics["mape"],
                        "overfit_gap": cv_summary["overfit_gap"],
                    }
                )
            except Exception as error:
                LOGGER.exception("Model comparison failed for %s: %s", model_name, error)
        comparison = pd.DataFrame(results)
        if comparison.empty:
            raise RuntimeError("No model completed successfully during comparison.")
        comparison = comparison.sort_values("test_rmse").reset_index(drop=True)
        comparison_path = resolve_path(Path(self.paths["reports_dir"]) / "model_comparison.csv")
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)
        return comparison
