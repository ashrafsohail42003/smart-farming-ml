from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.model import ModelTrainer
from src.utils import get_logger, load_artifact, load_config

LOGGER = get_logger("main")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop yield prediction pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "train", "tune", "compare", "stack", "evaluate"],
        default="full",
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "catboost", "xgboost", "ridge"],
        default="lgbm",
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data", type=str, default=None)
    return parser.parse_args()


def run_full_pipeline(trainer: ModelTrainer, model_name: str, n_trials: int) -> dict:
    baseline_metrics = trainer.train_baseline()
    cv_metrics = trainer.train_with_cv(model_name)
    trainer.tune_with_optuna(model_name, n_trials=n_trials)
    pipeline = trainer.train_final(model_name, use_best_params=True)
    test_metrics = trainer.evaluate(pipeline, model_name)
    return {
        "baseline": baseline_metrics,
        "cv": cv_metrics,
        "test": test_metrics,
    }


def run_train_only(trainer: ModelTrainer, model_name: str) -> dict:
    trainer.train_with_cv(model_name)
    pipeline = trainer.train_final(model_name, use_best_params=False)
    return trainer.evaluate(pipeline, model_name)


def run_tune_only(trainer: ModelTrainer, model_name: str, n_trials: int) -> dict:
    best_params = trainer.tune_with_optuna(model_name, n_trials=n_trials)
    pipeline = trainer.train_final(model_name, use_best_params=True)
    metrics = trainer.evaluate(pipeline, model_name)
    return {"best_params": best_params, "metrics": metrics}


def main() -> None:
    started_at = time.time()
    args = parse_args()
    cfg = load_config(args.config)
    trainer = ModelTrainer(cfg)
    trainer.load_data(args.data or cfg["data"]["processed_path"])

    if args.mode == "full":
        results = run_full_pipeline(trainer, args.model, args.trials or cfg["optuna"]["n_trials"])
    elif args.mode == "train":
        results = run_train_only(trainer, args.model)
    elif args.mode == "tune":
        results = run_tune_only(trainer, args.model, args.trials or cfg["optuna"]["n_trials"])
    elif args.mode == "compare":
        results = trainer.compare_models().to_dict(orient="records")
    elif args.mode == "stack":
        pipeline = trainer.train_stacking()
        results = trainer.evaluate(pipeline, "stacking")
    else:
        model_path = Path(cfg["paths"]["models_dir"]) / f"{args.model}_final_pipeline.pkl"
        pipeline = load_artifact(model_path)
        results = trainer.evaluate(pipeline, args.model)

    elapsed = time.time() - started_at
    LOGGER.info("Pipeline completed in %.2f seconds", elapsed)
    LOGGER.info("Results: %s", results)


if __name__ == "__main__":
    main()
