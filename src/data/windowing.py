"""
Windowing utilities for time-series data.

Creates sliding window input-output pairs for direct multi-step forecasting.
Handles dataset creation, saving, and loading of processed numpy arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.loader import load_raw_csv
from src.data.preprocessing import preprocess_data, split_data
from src.data.scaler import (
    create_scaler,
    fit_transform_scaler,
    transform_with_scaler,
    save_scaler,
    load_scaler,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_windows(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    target_idx: int = 3,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window input-output pairs for direct forecasting.

    Args:
        data: 2D array of shape (n_timesteps, n_features).
        window_size: Number of past time steps to use as input.
        horizon: Number of future steps to predict.
        target_idx: Column index of the target variable (default: 3 for 'close').
        max_samples: Maximum number of samples to create. If None, use all available.

    Returns:
        Tuple of (X, y) where:
            X has shape (n_samples, window_size, n_features)
            y has shape (n_samples, horizon)
    """
    n_samples = len(data) - window_size - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Insufficient data: {len(data)} rows for window={window_size}, "
            f"horizon={horizon}. Need at least {window_size + horizon} rows."
        )

    effective_samples = min(n_samples, max_samples) if max_samples else n_samples
    start_idx = n_samples - effective_samples

    X = np.zeros((effective_samples, window_size, data.shape[1]))
    y = np.zeros((effective_samples, horizon))

    for i in range(effective_samples):
        X[i] = data[start_idx + i : start_idx + i + window_size]
        y[i] = data[start_idx + i + window_size : start_idx + i + window_size + horizon, target_idx]

    logger.info(f"Created windows: X={X.shape}, y={y.shape}")
    return X, y


def create_dataset(
    file_path: Union[str, Path],
    output_dir: Union[str, Path],
    window_size: int,
    horizon: int,
    features: Optional[List[str]] = None,
    target: str = "close",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scaler_type: str = "standard",
    csv_columns: Optional[List[str]] = None,
    datetime_format: str = "%Y-%m-%d %H:%M",
    force_recreate: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Full pipeline: load CSV → preprocess → split → scale → window → save.

    Creates and saves train/val/test numpy arrays and the fitted scaler.

    Args:
        file_path: Path to the raw CSV file.
        output_dir: Directory to save processed arrays.
        window_size: Number of past time steps for input.
        horizon: Number of future steps to predict.
        features: Feature columns to use.
        target: Target column name.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        scaler_type: Type of scaler to use.
        csv_columns: Column names for the CSV.
        datetime_format: Datetime parsing format.
        force_recreate: If True, recreate even if files exist.

    Returns:
        Dictionary with keys: train_x, train_y, val_x, val_y, test_x, test_y.
    """
    output_dir = Path(output_dir)

    # Check if processed files already exist
    required_files = [
        "train_x.npy", "train_y.npy",
        "val_x.npy", "val_y.npy",
        "test_x.npy", "test_y.npy",
        "scaler.pkl",
    ]

    if not force_recreate and all((output_dir / f).exists() for f in required_files):
        logger.info(f"Processed data already exists at {output_dir}, loading...")
        return load_processed_data(output_dir)

    # Default features
    if features is None:
        features = ["open", "high", "low", "close", "volume"]

    # Determine target index
    target_idx = features.index(target)

    # Step 1: Load raw CSV
    df = load_raw_csv(file_path, columns=csv_columns, datetime_format=datetime_format)

    # Step 2: Preprocess
    df = preprocess_data(df, features=features)
    
    # Store raw datetime range
    raw_start_datetime = df.index[0]
    raw_end_datetime = df.index[-1]

    # Step 3: Time-based split
    train_df, val_df, test_df = split_data(
        df, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    
    # Store datetime indices for metadata (before scaling converts to numpy arrays)
    train_timestamps = train_df.index.to_numpy()
    val_timestamps = val_df.index.to_numpy()
    test_timestamps = test_df.index.to_numpy()

    # Step 4: Fit scaler on training data, transform all splits
    scaler = create_scaler(scaler_type)
    train_scaled = fit_transform_scaler(scaler, train_df.values)
    val_scaled = transform_with_scaler(scaler, val_df.values)
    test_scaled = transform_with_scaler(scaler, test_df.values)

    # Step 5: Create sliding windows (support limiting samples)
    # If `max_samples` is provided, treat it as a global cap across all splits
    def _possible_samples(arr: np.ndarray) -> int:
        return max(0, len(arr) - window_size - horizon + 1)

    n_train_possible = _possible_samples(train_scaled)
    n_val_possible = _possible_samples(val_scaled)
    n_test_possible = _possible_samples(test_scaled)
    
    # Helper function to calculate datetime ranges for windowed splits
    def _calculate_datetime_range(
        timestamps: np.ndarray,
        n_samples_created: int,
        n_possible: int,
    ) -> Tuple[str, str]:
        """Calculate start and end datetime for a windowed split.
        
        Args:
            timestamps: Original timestamps for the split.
            n_samples_created: Number of samples actually created.
            n_possible: Number of samples possible from this split.
            
        Returns:
            Tuple of (start_datetime, end_datetime) as ISO format strings.
        """
        if n_samples_created == 0:
            return None, None
        
        # Start index accounts for taking most recent samples
        start_idx = n_possible - n_samples_created
        
        # First sample uses timestamps from start_idx to start_idx + window_size - 1 (input)
        # and start_idx + window_size to start_idx + window_size + horizon - 1 (output)
        start_datetime = pd.Timestamp(timestamps[start_idx])
        
        # Last sample uses timestamps up to:
        # start_idx + n_samples_created - 1 + window_size + horizon - 1
        end_idx = start_idx + n_samples_created - 1 + window_size + horizon - 1
        end_datetime = pd.Timestamp(timestamps[end_idx])
        
        return start_datetime.isoformat(), end_datetime.isoformat()

    if max_samples is None:
        # previous behavior: cap per-split individually (i.e., no cap)
        train_x, train_y = create_windows(train_scaled, window_size, horizon, target_idx)
        val_x, val_y = create_windows(val_scaled, window_size, horizon, target_idx)
        test_x, test_y = create_windows(test_scaled, window_size, horizon, target_idx)
        
        logger.warning(
            f"max_samples not specified. Created {train_x.shape[0] + val_x.shape[0] + test_x.shape[0]} total samples."
        )
        
        # Calculate datetime ranges
        train_start_dt, train_end_dt = _calculate_datetime_range(
            train_timestamps, train_x.shape[0], n_train_possible
        )
        val_start_dt, val_end_dt = _calculate_datetime_range(
            val_timestamps, val_x.shape[0], n_val_possible
        )
        test_start_dt, test_end_dt = _calculate_datetime_range(
            test_timestamps, test_x.shape[0], n_test_possible
        )
        
        # Create metadata dictionary
        split_metadata = {
            "raw_start_datetime": pd.Timestamp(raw_start_datetime).isoformat(),
            "raw_end_datetime": pd.Timestamp(raw_end_datetime).isoformat(),
            "raw_total_timesteps": len(df),
            "features": features,
            "target": target,
            "window_size": window_size,
            "horizon": horizon,
            "max_samples": max_samples,
            "scaler_type": scaler_type,
            "splits": {
                "train": {
                    "n_samples": int(train_x.shape[0]),
                    "start_datetime": train_start_dt,
                    "end_datetime": train_end_dt,
                    "split_ratio": train_ratio,
                },
                "val": {
                    "n_samples": int(val_x.shape[0]),
                    "start_datetime": val_start_dt,
                    "end_datetime": val_end_dt,
                    "split_ratio": val_ratio,
                },
                "test": {
                    "n_samples": int(test_x.shape[0]),
                    "start_datetime": test_start_dt,
                    "end_datetime": test_end_dt,
                    "split_ratio": test_ratio,
                },
            },
            "total_windowed_samples": int(train_x.shape[0] + val_x.shape[0] + test_x.shape[0]),
        }
    else:
        total_possible = n_train_possible + n_val_possible + n_test_possible
        if total_possible == 0:
            raise ValueError(
                f"Insufficient data across splits for window={window_size}, horizon={horizon}."
            )

        desired_total = min(total_possible, int(max_samples))
        logger.info(
            f"Applying max_samples={max_samples}: possible={total_possible}, will create={desired_total}"
        )

        # Initial proportional allocation based on split ratios
        train_alloc = int(desired_total * train_ratio)
        val_alloc = int(desired_total * val_ratio)
        test_alloc = desired_total - train_alloc - val_alloc

        allocs = [train_alloc, val_alloc, test_alloc]
        possible = [n_train_possible, n_val_possible, n_test_possible]

        # Ensure allocations do not exceed available samples per split
        effective = [min(a, p) for a, p in zip(allocs, possible)]
        remaining = desired_total - sum(effective)

        # Distribute any remaining quota to splits that still have capacity
        idx = 0
        while remaining > 0:
            if possible[idx] - effective[idx] > 0:
                add = min(possible[idx] - effective[idx], remaining)
                effective[idx] += add
                remaining -= add
            idx = (idx + 1) % 3
            # If we've looped without being able to assign, break to avoid infinite loop
            if all(effective[i] == possible[i] for i in range(3)):
                break

        # Final per-split caps
        train_cap, val_cap, test_cap = effective

        train_x, train_y = create_windows(
            train_scaled, window_size, horizon, target_idx=target_idx, max_samples=train_cap
        ) if train_cap > 0 else (np.zeros((0, window_size, train_scaled.shape[1])), np.zeros((0, horizon)))

        val_x, val_y = create_windows(
            val_scaled, window_size, horizon, target_idx=target_idx, max_samples=val_cap
        ) if val_cap > 0 else (np.zeros((0, window_size, val_scaled.shape[1])), np.zeros((0, horizon)))

        test_x, test_y = create_windows(
            test_scaled, window_size, horizon, target_idx=target_idx, max_samples=test_cap
        ) if test_cap > 0 else (np.zeros((0, window_size, test_scaled.shape[1])), np.zeros((0, horizon)))

        # Enforce strict max_samples limit with assertion
        total_created = train_x.shape[0] + val_x.shape[0] + test_x.shape[0]
        assert total_created <= max_samples, (
            f"Dataset size {total_created} exceeds max_samples={max_samples}"
        )
        
        logger.info(
            f"[OK] max_samples={max_samples} enforced. Created: "
            f"Train={train_x.shape[0]}, Val={val_x.shape[0]}, Test={test_x.shape[0]}, "
            f"Total={total_created}"
        )

        # Calculate datetime ranges
        train_start_dt, train_end_dt = _calculate_datetime_range(
            train_timestamps, train_x.shape[0], n_train_possible
        )
        val_start_dt, val_end_dt = _calculate_datetime_range(
            val_timestamps, val_x.shape[0], n_val_possible
        )
        test_start_dt, test_end_dt = _calculate_datetime_range(
            test_timestamps, test_x.shape[0], n_test_possible
        )
        
        # Create metadata dictionary
        split_metadata = {
            "raw_start_datetime": pd.Timestamp(raw_start_datetime).isoformat(),
            "raw_end_datetime": pd.Timestamp(raw_end_datetime).isoformat(),
            "raw_total_timesteps": len(df),
            "features": features,
            "target": target,
            "window_size": window_size,
            "horizon": horizon,
            "max_samples": max_samples,
            "scaler_type": scaler_type,
            "splits": {
                "train": {
                    "n_samples": int(train_x.shape[0]),
                    "start_datetime": train_start_dt,
                    "end_datetime": train_end_dt,
                    "split_ratio": train_ratio,
                },
                "val": {
                    "n_samples": int(val_x.shape[0]),
                    "start_datetime": val_start_dt,
                    "end_datetime": val_end_dt,
                    "split_ratio": val_ratio,
                },
                "test": {
                    "n_samples": int(test_x.shape[0]),
                    "start_datetime": test_start_dt,
                    "end_datetime": test_end_dt,
                    "split_ratio": test_ratio,
                },
            },
            "total_windowed_samples": int(train_x.shape[0] + val_x.shape[0] + test_x.shape[0]),
        }

    # Step 6: Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_x.npy", train_x)
    np.save(output_dir / "train_y.npy", train_y)
    np.save(output_dir / "val_x.npy", val_x)
    np.save(output_dir / "val_y.npy", val_y)
    np.save(output_dir / "test_x.npy", test_x)
    np.save(output_dir / "test_y.npy", test_y)
    save_scaler(scaler, output_dir / "scaler.pkl")
    
    # Save metadata
    metadata_path = output_dir / "split_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(split_metadata, f, indent=2)

    logger.info(
        f"Dataset saved to {output_dir} | "
        f"Train: {train_x.shape[0]}, Val: {val_x.shape[0]}, Test: {test_x.shape[0]}"
    )
    logger.info(f"Split metadata saved to {metadata_path}")

    return {
        "train_x": train_x, "train_y": train_y,
        "val_x": val_x, "val_y": val_y,
        "test_x": test_x, "test_y": test_y,
    }


def load_processed_data(data_dir: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load previously processed numpy arrays.

    Args:
        data_dir: Directory containing the saved .npy files.

    Returns:
        Dictionary with train/val/test X and y arrays.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    data_dir = Path(data_dir)
    required = ["train_x.npy", "train_y.npy", "val_x.npy", "val_y.npy",
                 "test_x.npy", "test_y.npy"]
    for f in required:
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"Missing processed file: {data_dir / f}")

    return {
        "train_x": np.load(data_dir / "train_x.npy"),
        "train_y": np.load(data_dir / "train_y.npy"),
        "val_x": np.load(data_dir / "val_x.npy"),
        "val_y": np.load(data_dir / "val_y.npy"),
        "test_x": np.load(data_dir / "test_x.npy"),
        "test_y": np.load(data_dir / "test_y.npy"),
    }
