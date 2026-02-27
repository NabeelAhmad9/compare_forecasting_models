"""
Logging utilities for the forecasting project.

Provides a centralized logger factory that supports both
console and file logging with configurable formats.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Get or create a named logger with console and optional file handlers.

    Args:
        name: Logger name (typically module name).
        log_file: Optional path to a log file.
        level: Logging level (default: INFO).
        fmt: Log message format string.

    Returns:
        Configured logger instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def setup_experiment_logger(
    experiment_type: str,
    model_name: str,
    category: str,
    asset: str,
    horizon: int,
    logs_dir: Union[str, Path],
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger for a specific experiment run.

    Args:
        experiment_type: Type of experiment ('hpo' or 'final').
        model_name: Name of the model.
        category: Asset category.
        asset: Asset name.
        horizon: Forecasting horizon.
        logs_dir: Base logs directory.
        level: Logging level.

    Returns:
        Configured experiment logger.
    """
    log_name = f"{experiment_type}.{model_name}.{category}.{asset}.h{horizon}"
    log_file = (
        Path(logs_dir) / experiment_type / f"{model_name}_{category}_{asset}_{horizon}.log"
    )
    return get_logger(log_name, log_file=log_file, level=level)
