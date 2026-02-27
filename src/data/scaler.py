"""
Scaler utilities for feature normalization.

Provides functions to create, fit, transform, save, and load scalers
to ensure consistent scaling across train, val, and test sets.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping of scaler types to classes
SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def create_scaler(scaler_type: str = "standard") -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
    """Create a new scaler instance.

    Args:
        scaler_type: Type of scaler ('standard', 'minmax', 'robust').

    Returns:
        Scaler instance (unfitted).

    Raises:
        ValueError: If scaler_type is not recognized.
    """
    if scaler_type not in SCALER_MAP:
        raise ValueError(
            f"Unknown scaler type: {scaler_type}. "
            f"Choose from {list(SCALER_MAP.keys())}"
        )
    return SCALER_MAP[scaler_type]()


def fit_transform_scaler(
    scaler: Union[StandardScaler, MinMaxScaler, RobustScaler],
    data: np.ndarray,
) -> np.ndarray:
    """Fit the scaler on data and return the transformed data.

    Args:
        scaler: Scaler instance.
        data: 2D numpy array (n_samples, n_features).

    Returns:
        Scaled data array.
    """
    return scaler.fit_transform(data)


def transform_with_scaler(
    scaler: Union[StandardScaler, MinMaxScaler, RobustScaler],
    data: np.ndarray,
) -> np.ndarray:
    """Transform data using an already-fitted scaler.

    Args:
        scaler: Fitted scaler instance.
        data: 2D numpy array (n_samples, n_features).

    Returns:
        Scaled data array.
    """
    return scaler.transform(data)


def inverse_transform_scaler(
    scaler: Union[StandardScaler, MinMaxScaler, RobustScaler],
    data: np.ndarray,
    n_features: int,
    target_idx: int = 3,
) -> np.ndarray:
    """Inverse-transform predictions back to original scale.

    Since predictions may only contain the target column, we need to
    pad them to match the scaler's expected number of features.

    Args:
        scaler: Fitted scaler instance.
        data: Scaled predictions (n_samples,) or (n_samples, 1).
        n_features: Total number of features the scaler was fit on.
        target_idx: Index of the target column in the feature set.

    Returns:
        Inverse-transformed predictions.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Create a dummy array with zeros for all features
    dummy = np.zeros((data.shape[0], n_features))
    dummy[:, target_idx] = data[:, 0]

    # Inverse transform and extract target column
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, target_idx]


def save_scaler(
    scaler: Union[StandardScaler, MinMaxScaler, RobustScaler],
    path: Union[str, Path],
) -> None:
    """Save a fitted scaler to disk.

    Args:
        scaler: Fitted scaler instance.
        path: File path for saving (typically .pkl).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {path}")


def load_scaler(
    path: Union[str, Path],
) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
    """Load a fitted scaler from disk.

    Args:
        path: Path to the saved scaler file.

    Returns:
        Fitted scaler instance.

    Raises:
        FileNotFoundError: If the scaler file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded from {path}")
    return scaler
