"""
Metric aggregation across seeds and experiments.

Computes mean and standard deviation of metrics over multiple
random seeds for robust performance estimation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Metadata keys that should not be aggregated (non-metric fields)
_SKIP_KEYS = frozenset({
    "seed", "model", "category", "asset", "horizon",
    "epochs_trained", "n_parameters",
})


def aggregate_seed_metrics(
    metrics_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple seeds.

    Computes mean and standard deviation for each numeric metric.
    Skips non-numeric metadata fields (model name, seed, etc.).

    Args:
        metrics_list: List of metric dictionaries, one per seed.

    Returns:
        Dictionary mapping metric name to {mean, std, values}.
    """
    if not metrics_list:
        return {}

    aggregated: Dict[str, Dict[str, float]] = {}

    for name in metrics_list[0].keys():
        if name in _SKIP_KEYS:
            continue
        values = [m[name] for m in metrics_list if name in m]
        # Only aggregate numeric values
        try:
            float_values = [float(v) for v in values]
        except (ValueError, TypeError):
            continue
        aggregated[name] = {
            "mean": float(np.mean(float_values)),
            "std": float(np.std(float_values)),
            "values": float_values,
        }

    return aggregated


def save_aggregated_metrics(
    aggregated: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
) -> None:
    """Save aggregated metrics to a JSON file.

    Args:
        aggregated: Aggregated metrics dictionary.
        output_path: Path to save the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"Aggregated metrics saved to {output_path}")


def load_aggregated_metrics(
    path: Union[str, Path],
) -> Dict[str, Dict[str, float]]:
    """Load aggregated metrics from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Aggregated metrics dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Aggregated metrics not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_seed_metrics(
    results_dir: Union[str, Path],
    model_name: str,
    category: str,
    asset: str,
    horizon: int,
    seeds: List[int],
) -> List[Dict[str, float]]:
    """Load per-seed metrics from results directory.

    Args:
        results_dir: Base results directory.
        model_name: Model name.
        category: Asset category.
        asset: Asset name.
        horizon: Forecast horizon.
        seeds: List of seeds.

    Returns:
        List of metric dictionaries, one per seed.
    """
    results_dir = Path(results_dir)
    metrics_list = []

    for seed in seeds:
        base_path = (
            results_dir / "final" / model_name / category / asset / str(horizon) / str(seed)
        )
        # prefer best metrics, fall back to last
        candidates = [base_path / "best" / "metrics.json", base_path / "last" / "metrics.json"]
        metrics_path = None
        for cand in candidates:
            if cand.exists():
                metrics_path = cand
                break
        if metrics_path is None and (base_path / "metrics.json").exists():
            # in case older layout placed file directly
            metrics_path = base_path / "metrics.json"

        if metrics_path and metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_list.append(json.load(f))
        else:
            logger.warning(f"Missing metrics for seed {seed}: searched {candidates}")

    return metrics_list


def format_metric_string(mean: float, std: float, precision: int = 4) -> str:
    """Format a metric as 'mean ± std'.

    Args:
        mean: Mean value.
        std: Standard deviation.
        precision: Decimal places.

    Returns:
        Formatted string.
    """
    return f"{mean:.{precision}f} ± {std:.{precision}f}"
