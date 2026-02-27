"""
Final Evaluation Script — Best vs Last Checkpoints.

Evaluates both ``model_best.pt`` and ``model_last.pt`` checkpoints for every
trained model/category/asset/horizon/seed combination discovered under
``models/final/``.  Results are persisted with the structure::

    results/final/<model>/<category>/<asset>/<horizon>/<seed>/
        ├── best/
        │   ├── metrics.json
        │   └── predictions.csv
        └── last/
            ├── metrics.json
            └── predictions.csv

    results/final/<model>/<category>/<asset>/<horizon>/
        ├── aggregated_best.json
        └── aggregated_last.json

Usage::

    python final_evaluation.py              # evaluate ALL models
    python final_evaluation.py --model LSTM # evaluate a single model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.models import get_model_class, list_models
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.aggregate import aggregate_seed_metrics, save_aggregated_metrics
from src.data.scaler import load_scaler, inverse_transform_scaler
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE: int = 5                # OHLCV features
TARGET_IDX: int = 3                # "close" column index
EVAL_BATCH_SIZE: int = 256         # inference batch size
CHECKPOINT_TYPES: Tuple[str, ...] = ("best", "last")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FMT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"

logger = logging.getLogger("final_evaluation")


def _setup_logging(level: int = logging.INFO) -> None:
    """Configure root and script-level logging to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FMT))
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models" / "final"
RESULTS_DIR = ROOT_DIR / "results" / "final"


# ===================================================================
# 1.  Discovery helpers
# ===================================================================

def discover_models(models_dir: Path, model_filter: Optional[str] = None) -> List[str]:
    """Return sorted list of model names available under *models_dir*.

    Args:
        models_dir: ``models/final/`` directory.
        model_filter: If provided, return only this model (validated).

    Returns:
        Sorted model name list.
    """
    if not models_dir.exists():
        logger.warning("Models directory not found: %s", models_dir)
        return []

    if model_filter is not None:
        if (models_dir / model_filter).is_dir():
            return [model_filter]
        logger.warning("Model '%s' not found in %s", model_filter, models_dir)
        return []

    return sorted(
        d.name for d in models_dir.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    )


def _discover_subdirs(parent: Path) -> List[str]:
    """Return sorted child directory names (skip hidden / underscore)."""
    if not parent.is_dir():
        return []
    return sorted(
        d.name for d in parent.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    )


def discover_categories(model_dir: Path) -> List[str]:
    """Discover asset categories (crypto, forex, indices)."""
    return _discover_subdirs(model_dir)


def discover_assets(category_dir: Path) -> List[str]:
    """Discover asset names inside a category directory."""
    return _discover_subdirs(category_dir)


def discover_horizons(asset_dir: Path) -> List[int]:
    """Discover integer horizon values."""
    horizons: List[int] = []
    for name in _discover_subdirs(asset_dir):
        try:
            horizons.append(int(name))
        except ValueError:
            continue
    return sorted(horizons)


def discover_seeds(horizon_dir: Path) -> List[int]:
    """Discover integer seed values."""
    seeds: List[int] = []
    for name in _discover_subdirs(horizon_dir):
        try:
            seeds.append(int(name))
        except ValueError:
            continue
    return sorted(seeds)


# ===================================================================
# 2.  Data loading
# ===================================================================

def load_test_data(
    category: str,
    asset: str,
    horizon: int,
    data_dir: Path = DATA_DIR,
) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    """Load ``test_x.npy``, ``test_y.npy`` and (optionally) ``scaler.pkl``.

    Args:
        category: Asset category (crypto / forex / indices).
        asset: Asset ticker.
        horizon: Forecast horizon.
        data_dir: Root ``data/`` directory.

    Returns:
        ``(test_x, test_y, scaler)`` — scaler is ``None`` if missing.

    Raises:
        FileNotFoundError: If test arrays are absent.
    """
    data_path = data_dir / "processed" / category / asset / str(horizon)

    test_x_path = data_path / "test_x.npy"
    test_y_path = data_path / "test_y.npy"
    scaler_path = data_path / "scaler.pkl"

    if not test_x_path.exists():
        raise FileNotFoundError(f"Test X not found: {test_x_path}")
    if not test_y_path.exists():
        raise FileNotFoundError(f"Test Y not found: {test_y_path}")

    test_x = np.load(test_x_path)
    test_y = np.load(test_y_path)

    scaler = load_scaler(scaler_path) if scaler_path.exists() else None

    logger.info("Loaded test data: X=%s  Y=%s  scaler=%s",
                test_x.shape, test_y.shape, scaler is not None)
    return test_x, test_y, scaler


def create_test_dataloader(
    test_x: np.ndarray,
    test_y: np.ndarray,
    batch_size: int = EVAL_BATCH_SIZE,
) -> DataLoader:
    """Wrap numpy arrays into a deterministic ``DataLoader``.

    Args:
        test_x: Input windows  — shape ``(N, window_size, features)``.
        test_y: Target horizons — shape ``(N, horizon)``.
        batch_size: Batch size.

    Returns:
        PyTorch ``DataLoader`` (no shuffling, no drop-last).
    """
    x_tensor = torch.from_numpy(test_x).float()
    y_tensor = torch.from_numpy(test_y).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


# ===================================================================
# 3.  Model loading
# ===================================================================

def load_model_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    input_size: int,
    window_size: int,
    horizon: int,
    device: torch.device,
) -> Optional[torch.nn.Module]:
    """Instantiate a model and load weights from *checkpoint_path*.

    Architecture-specific hyper-parameters are read from the checkpoint's
    ``hparams`` dict so no external config is needed.

    Args:
        model_name: Registered model name (e.g. ``"LSTM"``).
        checkpoint_path: ``.pt`` checkpoint file.
        input_size: Number of input features (5 for OHLCV).
        window_size: Lookback length (inferred from test data).
        horizon: Forecast steps.
        device: Target torch device.

    Returns:
        Model in eval mode, or ``None`` on failure.
    """
    if not checkpoint_path.exists():
        logger.warning("Checkpoint not found: %s", checkpoint_path)
        return None

    try:
        model_cls = get_model_class(model_name)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
        hparams: Dict[str, Any] = checkpoint.get("hparams", {})

        # Separate base-class args from architecture kwargs
        base_keys = {"model_name", "input_size", "window_size", "horizon"}
        model_kwargs = {k: v for k, v in hparams.items() if k not in base_keys}

        model = model_cls(
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            **model_kwargs,
        )

        # Load state_dict with DataParallel prefix handling
        state_dict = checkpoint["model_state_dict"]
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            stripped = {
                k.replace("module.", "", 1) if k.startswith("module.") else k: v
                for k, v in state_dict.items()
            }
            model.load_state_dict(stripped)

        model.to(device).eval()
        logger.info("Loaded checkpoint: %s", checkpoint_path)
        return model

    except Exception:
        logger.error("Failed to load model from %s:\n%s",
                      checkpoint_path, traceback.format_exc())
        return None


# ===================================================================
# 4.  Inference
# ===================================================================

def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run deterministic forward passes over *dataloader*.

    Args:
        model: Model in eval mode.
        dataloader: Test ``DataLoader``.
        device: Computation device.

    Returns:
        ``(y_true, y_pred)`` numpy arrays.
    """
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            y_true_parts.append(batch_y.numpy())
            y_pred_parts.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_parts, axis=0)
    y_pred = np.concatenate(y_pred_parts, axis=0)
    return y_true, y_pred


# ===================================================================
# 5.  Inverse-transform helpers
# ===================================================================

def inverse_transform_predictions(
    y: np.ndarray,
    scaler: Any,
    n_features: int = INPUT_SIZE,
    target_idx: int = TARGET_IDX,
) -> np.ndarray:
    """Inverse-scale predictions / targets back to the original domain.

    Handles both 1-D ``(N,)`` and multi-step ``(N, horizon)`` arrays by
    delegating each horizon step to :func:`inverse_transform_scaler`.

    Args:
        y: Scaled values.
        scaler: Fitted scaler (e.g. ``StandardScaler``).
        n_features: Total feature count the scaler was fitted on.
        target_idx: Column index of the target feature.

    Returns:
        Array of the same shape in the original scale.
    """
    if y.ndim == 1:
        return inverse_transform_scaler(scaler, y, n_features, target_idx)

    # Multi-step: inverse-transform each horizon step independently
    inversed_steps = [
        inverse_transform_scaler(scaler, y[:, step], n_features, target_idx)
        for step in range(y.shape[1])
    ]
    return np.column_stack(inversed_steps)


# ===================================================================
# 6.  Saving helpers
# ===================================================================

def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Persist ground-truth and predicted values as a tidy CSV.

    For multi-step forecasts the columns follow the naming convention
    ``y_true_step_1 … y_true_step_H, y_pred_step_1 … y_pred_step_H``.

    Args:
        y_true: Ground truth array.
        y_pred: Prediction array.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if y_true.ndim == 1:
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    else:
        cols_true = [f"y_true_step_{i + 1}" for i in range(y_true.shape[1])]
        cols_pred = [f"y_pred_step_{i + 1}" for i in range(y_pred.shape[1])]
        df = pd.concat(
            [pd.DataFrame(y_true, columns=cols_true),
             pd.DataFrame(y_pred, columns=cols_pred)],
            axis=1,
        )

    df.to_csv(output_path, index=False)
    logger.debug("Predictions saved → %s", output_path)


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Write a metrics dictionary to a JSON file.

    Args:
        metrics: Metric name → value mapping.
        output_path: Destination JSON path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.debug("Metrics saved → %s", output_path)


# ===================================================================
# 7.  Unified evaluation function
# ===================================================================

def evaluate_checkpoint(
    model_name: str,
    category: str,
    asset: str,
    horizon: int,
    seed: int,
    checkpoint_type: str,
    test_x: np.ndarray,
    test_y: np.ndarray,
    scaler: Optional[Any],
    models_dir: Path,
    results_dir: Path,
    device: torch.device,
    batch_size: int = EVAL_BATCH_SIZE,
) -> Optional[Dict[str, Any]]:
    """Evaluate a single checkpoint and persist its results.

    Steps:
        1. Load the checkpoint (``model_best.pt`` or ``model_last.pt``).
        2. Run inference on the test set.
        3. Inverse-transform predictions and targets if a scaler exists.
        4. Compute all metrics.
        5. Save ``metrics.json`` and ``predictions.csv``.

    Args:
        model_name: Registered model name.
        category: Asset category.
        asset: Asset ticker.
        horizon: Forecast horizon.
        seed: Training seed.
        checkpoint_type: ``"best"`` or ``"last"``.
        test_x: Test input array.
        test_y: Test target array.
        scaler: Fitted scaler instance (or ``None``).
        models_dir: Root checkpoint directory (``models/final``).
        results_dir: Root results directory (``results/final``).
        device: Torch device.
        batch_size: Inference batch size.

    Returns:
        Metrics dictionary, or ``None`` if the checkpoint is missing /
        evaluation failed.
    """
    checkpoint_path = (
        models_dir / model_name / category / asset
        / str(horizon) / str(seed) / f"model_{checkpoint_type}.pt"
    )

    if not checkpoint_path.exists():
        logger.warning("Checkpoint missing — skipping: %s", checkpoint_path)
        return None

    # Infer window size from the test data itself
    window_size = test_x.shape[1]

    # --- load model ---
    model = load_model_checkpoint(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        input_size=INPUT_SIZE,
        window_size=window_size,
        horizon=horizon,
        device=device,
    )
    if model is None:
        return None

    # --- inference ---
    dataloader = create_test_dataloader(test_x, test_y, batch_size)
    y_true, y_pred = run_inference(model, dataloader, device)

    # --- inverse-transform ---
    if scaler is not None:
        y_true = inverse_transform_predictions(y_true, scaler)
        y_pred = inverse_transform_predictions(y_pred, scaler)

    # --- metrics ---
    metrics = compute_all_metrics(y_true, y_pred)
    metrics.update({
        "seed": seed,
        "model": model_name,
        "category": category,
        "asset": asset,
        "horizon": horizon,
        "n_parameters": model.count_parameters(),
    })

    # --- persist ---
    result_dir = (
        results_dir / model_name / category / asset
        / str(horizon) / str(seed) / checkpoint_type
    )
    save_metrics(metrics, result_dir / "metrics.json")
    save_predictions(y_true, y_pred, result_dir / "predictions.csv")

    logger.info(
        "[%s] %s/%s/%s/H%d/S%d — RMSE: %.4f  MAE: %.4f",
        checkpoint_type.upper(), model_name, category, asset,
        horizon, seed, metrics["rmse"], metrics["mae"],
    )
    return metrics


# ===================================================================
# 8.  Seed-level evaluation (both checkpoints)
# ===================================================================

def evaluate_seed(
    model_name: str,
    category: str,
    asset: str,
    horizon: int,
    seed: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
    scaler: Optional[Any],
    models_dir: Path,
    results_dir: Path,
    device: torch.device,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Evaluate both ``model_best.pt`` and ``model_last.pt`` for one seed.

    Args:
        model_name: Registered model name.
        category / asset / horizon / seed: Experiment identifiers.
        test_x / test_y / scaler: Pre-loaded test data.
        models_dir / results_dir: Root directories.
        device: Torch device.

    Returns:
        ``(metrics_best, metrics_last)`` — either may be ``None``.
    """
    common_kwargs = dict(
        model_name=model_name,
        category=category,
        asset=asset,
        horizon=horizon,
        seed=seed,
        test_x=test_x,
        test_y=test_y,
        scaler=scaler,
        models_dir=models_dir,
        results_dir=results_dir,
        device=device,
    )

    metrics_best = evaluate_checkpoint(checkpoint_type="best", **common_kwargs)
    metrics_last = evaluate_checkpoint(checkpoint_type="last", **common_kwargs)
    return metrics_best, metrics_last


# ===================================================================
# 9.  Aggregation across seeds
# ===================================================================

def aggregate_checkpoint_metrics(
    model_name: str,
    category: str,
    asset: str,
    horizon: int,
    checkpoint_type: str,
    results_dir: Path,
) -> None:
    """Read per-seed ``metrics.json`` files and create an aggregated summary.

    Args:
        model_name: Model architecture.
        category / asset / horizon: Experiment identifiers.
        checkpoint_type: ``"best"`` or ``"last"``.
        results_dir: Root results directory.
    """
    horizon_dir = results_dir / model_name / category / asset / str(horizon)
    if not horizon_dir.exists():
        logger.warning("No results directory: %s", horizon_dir)
        return

    metrics_list: List[Dict[str, Any]] = []
    for seed_dir in sorted(horizon_dir.iterdir()):
        if not seed_dir.is_dir() or seed_dir.name.startswith(("_", ".")):
            continue
        metrics_path = seed_dir / checkpoint_type / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as fh:
                metrics_list.append(json.load(fh))

    if not metrics_list:
        logger.warning(
            "No seed metrics for %s/%s/%s/H%d/%s",
            model_name, category, asset, horizon, checkpoint_type,
        )
        return

    aggregated = aggregate_seed_metrics(metrics_list)
    output_path = horizon_dir / f"aggregated_{checkpoint_type}.json"
    save_aggregated_metrics(aggregated, output_path)

    logger.info(
        "Aggregated %d seed(s) → %s/%s/%s/H%d [%s]",
        len(metrics_list), model_name, category, asset, horizon,
        checkpoint_type,
    )


# ===================================================================
# 10. Orchestration
# ===================================================================

def run_full_evaluation(
    model_filter: Optional[str] = None,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    data_dir: Path = DATA_DIR,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Top-level driver: discover → evaluate → aggregate.

    Args:
        model_filter: Evaluate only this model, or ``None`` for all.
        models_dir: ``models/final/`` root.
        results_dir: ``results/final/`` root.
        data_dir: ``data/`` root.
        device: Torch device (auto-detected when ``None``).

    Returns:
        Summary dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    models = discover_models(models_dir, model_filter)
    if not models:
        logger.error("No models found to evaluate.")
        return {}

    logger.info("Models to evaluate (%d): %s", len(models), models)

    summary: Dict[str, Any] = {
        "total_seeds": 0,
        "successful_best": 0,
        "successful_last": 0,
        "failed_best": 0,
        "failed_last": 0,
        "models_evaluated": [],
    }

    for model_name in tqdm(models, desc="Models", unit="model"):
        model_dir = models_dir / model_name

        for category in discover_categories(model_dir):
            for asset in discover_assets(model_dir / category):
                for horizon in discover_horizons(model_dir / category / asset):

                    # Load test data once per (category, asset, horizon)
                    try:
                        test_x, test_y, scaler = load_test_data(
                            category, asset, horizon, data_dir,
                        )
                    except FileNotFoundError as exc:
                        logger.error("Missing test data — skipping: %s", exc)
                        continue

                    horizon_dir = model_dir / category / asset / str(horizon)
                    seeds = discover_seeds(horizon_dir)

                    best_collected: List[Dict[str, Any]] = []
                    last_collected: List[Dict[str, Any]] = []

                    desc = f"{model_name}/{category}/{asset}/H{horizon}"
                    for seed in tqdm(seeds, desc=desc, leave=False, unit="seed"):
                        summary["total_seeds"] += 1

                        m_best, m_last = evaluate_seed(
                            model_name=model_name,
                            category=category,
                            asset=asset,
                            horizon=horizon,
                            seed=seed,
                            test_x=test_x,
                            test_y=test_y,
                            scaler=scaler,
                            models_dir=models_dir,
                            results_dir=results_dir,
                            device=device,
                        )

                        if m_best is not None:
                            summary["successful_best"] += 1
                            best_collected.append(m_best)
                        else:
                            summary["failed_best"] += 1

                        if m_last is not None:
                            summary["successful_last"] += 1
                            last_collected.append(m_last)
                        else:
                            summary["failed_last"] += 1

                    # Aggregate across seeds for each checkpoint type
                    for ckpt_type, collected in [
                        ("best", best_collected),
                        ("last", last_collected),
                    ]:
                        if collected:
                            aggregate_checkpoint_metrics(
                                model_name=model_name,
                                category=category,
                                asset=asset,
                                horizon=horizon,
                                checkpoint_type=ckpt_type,
                                results_dir=results_dir,
                            )

        summary["models_evaluated"].append(model_name)

    # --- summary ---
    logger.info("=" * 72)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 72)
    logger.info("Models evaluated : %s", ", ".join(summary["models_evaluated"]))
    logger.info("Total seeds      : %d", summary["total_seeds"])
    logger.info("Successful (best): %d", summary["successful_best"])
    logger.info("Successful (last): %d", summary["successful_last"])
    logger.info("Failed     (best): %d", summary["failed_best"])
    logger.info("Failed     (last): %d", summary["failed_last"])
    logger.info("=" * 72)

    return summary


# ===================================================================
# 11. CLI / entry-point
# ===================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model_best.pt and model_last.pt checkpoints.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluate only this model (default: all discovered models).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EVAL_BATCH_SIZE,
        help=f"Inference batch size (default: {EVAL_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _setup_logging()

    args = parse_args()

    # Determinism
    set_seed(args.seed, deterministic=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_full_evaluation(
        model_filter=args.model,
        models_dir=MODELS_DIR,
        results_dir=RESULTS_DIR,
        data_dir=DATA_DIR,
        device=device,
    )
