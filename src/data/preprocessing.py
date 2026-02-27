"""
Data preprocessing utilities.

Handles data cleaning, feature selection, and time-based splitting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_data(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Preprocess raw OHLCV data.

    Steps:
        1. Select feature columns.
        2. Drop rows with any NaN values.
        3. Sort by datetime index.

    Args:
        df: Raw DataFrame with datetime index and OHLCV columns.
        features: List of feature columns to keep. Defaults to OHLCV.

    Returns:
        Cleaned DataFrame with selected features.
    """
    if features is None:
        features = ["open", "high", "low", "close", "volume"]

    # Ensure all requested features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing}")

    df = df[features].copy()

    # Remove NaN rows
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with NaN values")

    # Sort by index (datetime)
    df = df.sort_index()

    logger.info(f"Preprocessed data: {len(df)} rows, {len(features)} features")
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train, validation, and test sets.

    Uses time-based splitting to prevent data leakage.

    Args:
        df: Preprocessed DataFrame sorted by datetime.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios don't sum to approximately 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    return train_df, val_df, test_df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """Generate a summary of the dataset.

    Args:
        df: DataFrame to summarize.

    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "n_rows": len(df),
        "n_features": len(df.columns),
        "features": list(df.columns),
        "date_range": {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "missing_values": df.isnull().sum().to_dict(),
        "statistics": {},
    }

    for col in df.columns:
        summary["statistics"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }

    return summary
