"""
Evaluation metrics for time-series forecasting.

Implements RMSE, MAE, Directional Accuracy, and MSE metrics
used for model comparison and benchmarking.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial.distance import cdist

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MSE value.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))

def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE is scale-independent and more stable than MAPE,
    especially when values are near zero.

    Formula:
        SMAPE = 100 * mean( 2 * |y_pred - y_true| / (|y_true| + |y_pred| + epsilon) )

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Small constant to avoid division by zero.

    Returns:
        SMAPE value as percentage (0–100).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    numerator = 2.0 * np.abs(y_pred - y_true)

    return float(100.0 * np.mean(numerator / denominator))

def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute classic Dynamic Time Warping (DTW) distance between two sequences.
    
    Args:
        x: Sequence of shape (T1,)
        y: Sequence of shape (T2,)
    
    Returns:
        DTW distance (float)
    """
    x = np.array(x, dtype=np.float64).reshape(-1, 1)
    y = np.array(y, dtype=np.float64).reshape(-1, 1)
    
    D = cdist(x, y, metric="sqeuclidean")
    N, M = D.shape
    R = np.full((N + 1, M + 1), np.inf)
    R[0, 0] = 0.0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = D[i - 1, j - 1]
            R[i, j] = cost + min(R[i - 1, j], R[i, j - 1], R[i - 1, j - 1])
    
    return float(R[N, M])


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    last_known: np.ndarray | None = None,
) -> float:
    """Directional Accuracy (percentage).

    Measures the percentage of time steps where the predicted direction
    of change matches the actual direction. Uses step-by-step direction
    within each forecast horizon, and optionally the last known value
    for the first step direction.

    Args:
        y_true: Ground truth values (n_samples, horizon).
        y_pred: Predicted values (n_samples, horizon).
        last_known: Optional last known values (n_samples,) for 
            computing direction of the first forecast step.

    Returns:
        Directional accuracy as a percentage (0-100).
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    total_correct = 0
    total_count = 0

    # Direction from last known value to first forecast step
    if last_known is not None and y_true.shape[1] >= 1:
        true_dir_first = y_true[:, 0] > last_known
        pred_dir_first = y_pred[:, 0] > last_known
        total_correct += np.sum(true_dir_first == pred_dir_first)
        total_count += len(last_known)

    # Step-by-step direction within horizon
    if y_true.shape[1] >= 2:
        true_direction = np.diff(y_true, axis=1) > 0
        pred_direction = np.diff(y_pred, axis=1) > 0
        total_correct += np.sum(true_direction == pred_direction)
        total_count += true_direction.size

    if total_count == 0:
        return 50.0  # No directional information

    return float(total_correct / total_count * 100)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with RMSE, MAE, MSE, and Directional Accuracy.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "smape": smape(y_true, y_pred), 
        # "dtw": dtw_distance(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
