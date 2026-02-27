"""
Hyperparameter Optimization runner.

Fixed-seed HPO using Optuna on representative assets per asset class.
Saves best configurations for downstream multi-seed training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import pandas as pd

from src.data.windowing import load_processed_data
from src.models import get_model_class
from src.training.trainer import Trainer
from src.utils.config import ProjectConfig, save_yaml
from src.utils.logger import get_logger, setup_experiment_logger
from src.utils.seed import set_seed, get_device

logger = get_logger(__name__)

SAMPLER_MAP = {
    "TPE": TPESampler,
    "Random": RandomSampler,
    "CmaEs": CmaEsSampler,
}

PRUNER_MAP = {
    "median": MedianPruner,
    "hyperband": HyperbandPruner,
}


def _suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    """Suggest a hyperparameter value based on search space specification.

    Args:
        trial: Optuna trial.
        name: Parameter name.
        spec: Search space specification dict with 'type' and bounds.

    Returns:
        Suggested value.
    """
    param_type = spec["type"]

    if param_type == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    elif param_type == "float":
        step = spec.get("step", None)
        return trial.suggest_float(name, spec["low"], spec["high"], step=step)
    elif param_type == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    elif param_type == "categorical":
        choices = spec["choices"]
        # Handle nested lists (e.g., n_pool_kernel_size)
        if choices and isinstance(choices[0], list):
            idx = trial.suggest_categorical(name + "_idx", list(range(len(choices))))
            return choices[idx]
        return trial.suggest_categorical(name, choices)
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def run_hpo(
    config: ProjectConfig,
    model_name: str,
    category: str,
    horizon: int,
    models_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run HPO for a single model-category-horizon combination.

    Uses the representative asset for the category.

    Args:
        config: Project configuration.
        model_name: Name of the model to tune.
        category: Asset category.
        horizon: Forecast horizon.
        models_filter: Optional filter (unused, for API compatibility).

    Returns:
        Dictionary with best parameters and trial results.
    """
    # Setup
    fixed_seed = config.hpo.get("fixed_seed", 42)
    set_seed(fixed_seed)
    device = get_device(config.base.get("device", "auto"))

    representative = config.get_representative_asset(category)

    exp_logger = setup_experiment_logger(
        "hpo", model_name, category, representative, horizon,
        config.get_path("logs_dir"),
    )
    exp_logger.info(
        f"Starting HPO: model={model_name}, category={category}, "
        f"asset={representative}, horizon={horizon}"
    )

    # Prepare data path
    data_dir = config.get_path("data_processed") / category / representative / str(horizon)

    # Load preprocessed data
    exp_logger.info(f"Loading processed data from {data_dir}")
    dataset = load_processed_data(data_dir)

    # Get paired window size for this horizon (needed for model initialization)
    window_size = config.get_window_size(horizon)

    # Load search space
    search_space_config = config.get_model_search_space(model_name)
    search_space = search_space_config["search_space"]


    # Checkpoint directory for HPO trials
    hpo_models_dir = (
        config.get_path("models_dir") / "hpo" / model_name / category
        / representative / str(horizon)
    )

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Suggest hyperparameters
        hparams = {}
        for name, spec in search_space.items():
            hparams[name] = _suggest_param(trial, name, spec)

        # Early validation for model-specific constraints (fail-fast during HPO)
        if model_name == "ModernTCN":
            ps = hparams.get("patch_size")
            pst = hparams.get("patch_stride")
            if ps is not None and pst is not None and pst > ps:
                # Mark this trial as pruned — invalid hyperparameter combination
                raise optuna.exceptions.TrialPruned(
                    f"Invalid ModernTCN config: patch_stride ({pst}) > patch_size ({ps})"
                )

        # Extract training-level params
        trial_lr = hparams.pop("learning_rate", config.hpo.get("learning_rate", 0.001))
        trial_bs = hparams.pop("batch_size", config.hpo.get("batch_size", 64))

        # Special handling for ModernTCN arch_preset
        if model_name == "ModernTCN" and "arch_preset" in hparams:
            preset = hparams.pop("arch_preset")
            # Map preset index to architecture parameters
            arch_configs = {
                0: {  # 2-stage small
                    "num_blocks": [2, 2],
                    "dims": [64, 128],
                    "large_size": [31, 29],
                    "small_size": [5, 5],
                    "dw_dims": [64, 128],
                },
                1: {  # 2-stage medium
                    "num_blocks": [3, 3],
                    "dims": [64, 128],
                    "large_size": [31, 29],
                    "small_size": [7, 7],
                    "dw_dims": [64, 128],
                },
                2: {  # 2-stage large
                    "num_blocks": [2, 2],
                    "dims": [128, 256],
                    "large_size": [31, 29],
                    "small_size": [5, 5],
                    "dw_dims": [128, 256],
                },
                3: {  # 3-stage small
                    "num_blocks": [2, 2, 2],
                    "dims": [64, 128, 256],
                    "large_size": [51, 49, 47],
                    "small_size": [5, 5, 5],
                    "dw_dims": [64, 128, 256],
                },
                4: {  # 3-stage large
                    "num_blocks": [3, 3, 3],
                    "dims": [64, 128, 256],
                    "large_size": [51, 49, 47],
                    "small_size": [7, 7, 7],
                    "dw_dims": [64, 128, 256],
                },
                5: {  # 1-stage (multi-scale compatible)
                    "num_blocks": [3],
                    "dims": [128],
                    "large_size": [51],
                    "small_size": [7],
                    "dw_dims": [128],
                },
            }
            arch_config = arch_configs[preset]
            # Set use_multi_scale based on preset (only for single-stage)
            if "use_multi_scale" not in hparams:
                arch_config["use_multi_scale"] = (preset == 5)
            hparams.update(arch_config)

        # Build model
        model_cls = get_model_class(model_name)
        model = model_cls(
            input_size=len(config.dataset["features"]),
            window_size=window_size,
            horizon=horizon,
            **hparams,
        )

        # Training config for this trial
        trial_config = {
            **config.training,
            "epochs": config.hpo.get("epochs", 50),
            "learning_rate": trial_lr,
            "batch_size": trial_bs,
        }

        trial_checkpoint_dir = hpo_models_dir / f"trial_{trial.number:03d}"
        
        # Prepare hyperparameters dict for resume tracking
        all_hparams = {**hparams, "learning_rate": trial_lr, "batch_size": trial_bs}

        trainer = Trainer(model, device, trial_config, seed=fixed_seed)

        try:
            results = trainer.fit(
                train_x=dataset["train_x"],
                train_y=dataset["train_y"],
                val_x=dataset["val_x"],
                val_y=dataset["val_y"],
                checkpoint_dir=trial_checkpoint_dir,
                resume=True,
                model_name=model_name,
                category=category,
                asset=representative,
                horizon=horizon,
                hyperparameters=all_hparams,
            )
            return results["best_val_loss"]
        except Exception as e:
            exp_logger.error(f"Trial {trial.number} failed: {e}")
            return float("inf")

    # Check if we can resume from saved trials to avoid re-running completed HPO
    trials_csv = hpo_models_dir / "trials.csv"
    best_trial_json = hpo_models_dir / "best_trial.json"
    
    if trials_csv.exists() and best_trial_json.exists():
        # Load existing results to check completion
        with open(best_trial_json, "r") as f:
            best_result = json.load(f)
        
        trials_df = pd.read_csv(trials_csv)
        completed_trials = len(trials_df[trials_df["state"] == "TrialState.COMPLETE"])
        n_trials = config.hpo.get("n_trials", 5)
        
        if completed_trials >= n_trials:
            exp_logger.info(
                f"HPO already complete: {completed_trials}/{n_trials} trials finished"
            )
            # Return best result cached from previous run
            return best_result
        
        exp_logger.info(
            f"Resuming HPO: {completed_trials}/{n_trials} trials completed"
        )
    
    # Create Optuna study (in-memory only, no SQLite database)
    sampler_cls = SAMPLER_MAP.get(config.hpo.get("sampler", "TPE"), TPESampler)
    sampler = sampler_cls(seed=fixed_seed)

    pruner_config = config.hpo.get("pruner", {})
    pruner_type = pruner_config.get("type", "median")
    pruner_cls = PRUNER_MAP.get(pruner_type, MedianPruner)
    pruner_kwargs = {}
    if "n_startup_trials" in pruner_config:
        pruner_kwargs["n_startup_trials"] = pruner_config["n_startup_trials"]
    if "n_warmup_steps" in pruner_config:
        pruner_kwargs["n_warmup_steps"] = pruner_config["n_warmup_steps"]
    pruner = pruner_cls(**pruner_kwargs)

    study = optuna.create_study(
        direction=config.hpo.get("direction", "minimize"),
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization
    n_trials = config.hpo.get("n_trials", 5)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Extract results
    best_trial = study.best_trial
    best_params = best_trial.params

    # Clean up categorical index params
    clean_params = {}
    for key, value in best_params.items():
        if key.endswith("_idx"):
            original_key = key[:-4]
            spec = search_space[original_key]
            clean_params[original_key] = spec["choices"][value]
        else:
            clean_params[key] = value

    exp_logger.info(f"Best trial: {best_trial.number}, value: {best_trial.value:.6f}")
    exp_logger.info(f"Best params: {clean_params}")

    # Save results
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "number": trial.number,
            "value": trial.value,
            "state": str(trial.state),
            "params": trial.params,
        })

    trials_df = pd.DataFrame([
        {"number": t["number"], "value": t["value"], "state": t["state"],
         **t["params"]}
        for t in trials_data
    ])
    trials_df.to_csv(hpo_models_dir / "trials.csv", index=False)

    best_result = {
        "trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": clean_params,
        "model_name": model_name,
        "category": category,
        "asset": representative,
        "horizon": horizon,
    }

    with open(hpo_models_dir / "best_trial.json", "w") as f:
        json.dump(best_result, f, indent=2)

    # Save best config for the category
    config.save_model_best_config(model_name, clean_params, category=category, horizon=horizon)

    exp_logger.info(f"HPO complete. Results saved to {hpo_models_dir}")

    return best_result


def run_all_hpo(config: ProjectConfig, models_filter: Optional[List[str]] = None) -> None:
    """Run HPO for all models, categories, and horizons.

    Args:
        config: Project configuration.
        models_filter: Optional list of model names to run (runs all if None).
    """
    models = models_filter or config.get_available_models()
    categories = config.get_categories()
    horizons = config.get_horizons()

    total = len(models) * len(categories) * len(horizons)
    completed = 0

    logger.info(f"Starting HPO: {total} combinations")

    for model_name in models:
        for category in categories:
            for horizon in horizons:
                completed += 1
                logger.info(
                    f"[{completed}/{total}] HPO: {model_name} / {category} / h={horizon}"
                )
                try:
                    run_hpo(config, model_name, category, horizon)
                except Exception as e:
                    logger.error(
                        f"HPO failed for {model_name}/{category}/h{horizon}: {e}"
                    )

    logger.info("All HPO runs complete")
