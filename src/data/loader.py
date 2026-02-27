"""
Data loading utilities.

Handles reading raw CSV files and resolving asset file paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_csv(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    datetime_format: str = "%Y-%m-%d %H:%M",
) -> pd.DataFrame:
    """Load a raw OHLCV CSV file into a DataFrame.

    The CSV files have no header row. Columns are assumed to be:
    datetime, open, high, low, close, volume.

    Args:
        file_path: Path to the CSV file.
        columns: Column names to assign. Defaults to OHLCV standard.
        datetime_format: Format string for parsing datetime column.

    Returns:
        DataFrame with parsed datetime index and OHLCV columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV cannot be parsed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    if columns is None:
        columns = ["datetime", "open", "high", "low", "close", "volume"]

    logger.info(f"Loading CSV: {file_path}")

    df = pd.read_csv(
        file_path,
        header=None,
        names=columns,
        parse_dates=["datetime"],
        date_format=datetime_format,
    )

    # Sort by datetime and reset index
    df = df.sort_values("datetime").reset_index(drop=True)

    # Set datetime as index
    df = df.set_index("datetime")

    logger.info(f"Loaded {len(df)} rows from {file_path.name}")
    return df


def get_asset_file_path(
    data_raw_dir: Union[str, Path],
    asset_file: str,
) -> Path:
    """Resolve the full path to a raw asset CSV file.

    Args:
        data_raw_dir: Directory containing raw CSV files.
        asset_file: Filename of the asset CSV (e.g., 'BTCUSDT_H1.csv').

    Returns:
        Full path to the asset file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(data_raw_dir) / asset_file
    if not path.exists():
        raise FileNotFoundError(f"Asset file not found: {path}")
    return path


def list_available_assets(data_raw_dir: Union[str, Path]) -> List[str]:
    """List all available CSV files in the raw data directory.

    Args:
        data_raw_dir: Directory containing raw CSV files.

    Returns:
        Sorted list of CSV filenames.
    """
    raw_dir = Path(data_raw_dir)
    if not raw_dir.exists():
        return []
    return sorted(f.name for f in raw_dir.glob("*.csv"))
