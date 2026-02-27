"""
Final training runner.

Multi-seed retraining using frozen best hyperparameters from HPO.
Evaluates on test set and saves predictions and metrics per seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.data.windowing import load_processed_data
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.aggregate import aggregate_seed_metrics, save_aggregated_metrics
from src.models import get_model_class
from src.training.trainer import Trainer
from src.utils.config import ProjectConfig
from src.utils.logger import get_logger, setup_experiment_logger
from src.utils.seed import set_seed, get_device

logger = get_logger(__name__)


def run_final_single(
    config: ProjectConfig,
    model_name: str,
    category: str,
    asset_name: str,
    asset_file: str,
    horizon: int,
    seed: int,
) -> Dict[str, Any]:
    """Run final training for a single model-asset-horizon-seed combination.

    Args:
        config: Project configuration.
        model_name: Model architecture name.
        category: Asset category.
        asset_name: Asset name.
        asset_file: Asset CSV filename.
        horizon: Forecast horizon.
        seed: Random seed.

    Returns:
        Dictionary containing metrics and training metadata.
    """
    set_seed(seed)
    device = get_device(config.base.get("device", "auto"))

    exp_logger = setup_experiment_logger(
        "final", model_name, category, asset_name, horizon,
        config.get_path("logs_dir"),
    )
    exp_logger.info(
        f"Final training: model={model_name}, asset={asset_name}, "
        f"horizon={horizon}, seed={seed}"
    )

    # Prepare data path
    data_dir = config.get_path("data_processed") / category / asset_name / str(horizon)

    # Load preprocessed data
    exp_logger.info(f"Loading processed data from {data_dir}")
    dataset = load_processed_data(data_dir)


    # Get paired window size for this horizon (needed for model initialization)
    window_size = config.get_window_size(horizon)

    # Load best hyperparameters for this model and category
    try:
        best_hparams_raw = config.get_model_best_config(model_name, category=category, horizon=horizon)
        best_hparams = best_hparams_raw.get("best_params", best_hparams_raw)
    except FileNotFoundError:
        exp_logger.warning(
            f"No best config found for {model_name}/{category}/{horizon}. "
            f"Using default hyperparameters."
        )
        best_hparams = {}

    # Extract training-level params from best_hparams
    model_hparams = {k: v for k, v in best_hparams.items()
                     if k not in ("learning_rate", "batch_size")}
    train_overrides = {
        "learning_rate": best_hparams.get("learning_rate", config.training["learning_rate"]),
        "batch_size": best_hparams.get("batch_size", config.training["batch_size"]),
    }

    # Build model
    model_cls = get_model_class(model_name)
    model = model_cls(
        input_size=len(config.dataset["features"]),
        window_size=window_size,
        horizon=horizon,
        **model_hparams,
    )

    # Training config
    train_config = {**config.training, **train_overrides}

    # Checkpoint directory
    checkpoint_dir = (
        config.get_path("models_dir") /  "final" / model_name / category / asset_name
        / str(horizon) / str(seed)
    )

    # Results directory
    results_dir = (
        config.get_path("results_dir") / "final" / model_name / category
        / asset_name / str(horizon) / str(seed)
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        exp_logger.info(f"Results already exist at {metrics_path}, skipping seed {seed}...")
        with open(metrics_path, "r") as f:
            return json.load(f)

    # Train
    exp_logger.info(f"Setting up trainer for {model_name} on {asset_name} (H={horizon}, S={seed})...")
    trainer = Trainer(model, device, train_config, seed=seed)

    # Log hardware / runtime configuration for experiment-level visibility
    exp_logger.info(
        "Trainer runtime: device=%s | amp=%s | dp=%s | ddp=%s | compile=%s",
        trainer.device,
        getattr(trainer, "use_amp", False),
        getattr(trainer, "using_dp", False),
        getattr(trainer, "using_ddp", False),
        getattr(trainer, "use_compile", False),
    )

    exp_logger.info("Executing trainer.fit()...")
    train_results = trainer.fit(
        train_x=dataset["train_x"],
        train_y=dataset["train_y"],
        val_x=dataset["val_x"],
        val_y=dataset["val_y"],
        checkpoint_dir=checkpoint_dir,
        resume=config.training.get("checkpoint", {}).get("resume", True),
        model_name=model_name,
        category=category,
        asset=asset_name,
        horizon=horizon,
        hyperparameters=best_hparams,
    )

    # Evaluate on test set
    exp_logger.info("Generating predictions on test set...")
    predictions = trainer.predict(dataset["test_x"])
    exp_logger.info("Predictions generated successfully.")

    # Compute metrics (in scaled space)
    metrics = compute_all_metrics(dataset["test_y"], predictions)

    exp_logger.info(
        f"Test metrics - RMSE: {metrics['rmse']:.6f}, "
        f"MAE: {metrics['mae']:.6f}, DA: {metrics['directional_accuracy']:.2f}%"
    )

    # Save metrics
    metrics["seed"] = seed
    metrics["model"] = model_name
    metrics["category"] = category
    metrics["asset"] = asset_name
    metrics["horizon"] = horizon
    metrics["epochs_trained"] = train_results["epochs_trained"]
    metrics["n_parameters"] = train_results["n_parameters"]
    metrics["best_val_loss"] = train_results["best_val_loss"]

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame(predictions, columns=[f"h{i+1}" for i in range(horizon)])
    pred_df.to_csv(results_dir / "predictions.csv", index=False)

    exp_logger.info(f"Results saved to {results_dir}")

    return metrics


def run_final_for_model_asset(
    config: ProjectConfig,
    model_name: str,
    category: str,
    asset_name: str,
    asset_file: str,
    horizon: int,
) -> Dict[str, Any]:
    """Run final training for all seeds for a model-asset-horizon combo.

    Args:
        config: Project configuration.
        model_name: Model name.
        category: Asset category.
        asset_name: Asset name.
        asset_file: CSV filename.
        horizon: Forecast horizon.

    Returns:
        Aggregated metrics across seeds.
    """
    seeds = config.get_eval_seeds()
    all_metrics = []

    for seed in seeds:
        try:
            metrics = run_final_single(
                config, model_name, category, asset_name, asset_file, horizon, seed
            )
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(
                f"Final training failed: {model_name}/{asset_name}/h{horizon}/s{seed}: {e}"
            )

    # Aggregate metrics
    if all_metrics:
        aggregated = aggregate_seed_metrics(all_metrics)
        agg_path = (
            config.get_path("results_dir") / "final" / model_name / category
            / asset_name / str(horizon) / "aggregated.json"
        )
        save_aggregated_metrics(aggregated, agg_path)
        return aggregated

    return {}


def run_all_final(config: ProjectConfig, models_filter: Optional[List[str]] = None) -> None:
    """Run final training for all models, assets, and horizons.

    Args:
        config: Project configuration.
        models_filter: Optional list of model names (runs all if None).
    """
    models = models_filter or config.get_available_models()
    categories = config.get_categories()
    horizons = config.get_horizons()

    total_combos = sum(
        len(config.get_assets_for_category(cat))
        for cat in categories
    ) * len(models) * len(horizons)

    completed = 0

    logger.info(f"Starting final training: {total_combos} model-asset-horizon combinations")

    for model_name in models:
        for category in categories:
            assets = config.get_assets_for_category(category)
            for asset in assets:
                for horizon in horizons:
                    completed += 1
                    logger.info(
                        f"[{completed}/{total_combos}] Final: "
                        f"{model_name}/{asset['name']}/h={horizon}"
                    )
                    try:
                        run_final_for_model_asset(
                            config, model_name, category,
                            asset["name"], asset["file"], horizon,
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed: {model_name}/{asset['name']}/h{horizon}: {e}"
                        )

    logger.info("All final training complete")