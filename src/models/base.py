"""
Base forecaster class.

Defines the common interface that all forecasting models must implement.
Provides shared functionality for saving/loading checkpoints.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseForecaster(nn.Module, ABC):
    """Abstract base class for all forecasting models.

    All model architectures must inherit from this class and implement
    the forward method. Provides checkpoint save/load for resume support.

    Attributes:
        model_name: Name identifier for the model.
        input_size: Number of input features.
        window_size: Length of the input sequence (lookback).
        horizon: Number of future steps to forecast.
        hparams: Dictionary of model hyperparameters.
    """

    def __init__(
        self,
        model_name: str,
        input_size: int,
        window_size: int,
        horizon: int,
        **kwargs: Any,
    ) -> None:
        """Initialize the base forecaster.

        Args:
            model_name: Name of the model architecture.
            input_size: Number of input features (e.g., 5 for OHLCV).
            window_size: Length of input sequence.
            horizon: Number of steps to forecast.
            **kwargs: Additional hyperparameters stored in self.hparams.
        """
        super().__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.window_size = window_size
        self.horizon = horizon
        self.hparams = {
            "model_name": model_name,
            "input_size": input_size,
            "window_size": window_size,
            "horizon": horizon,
            **kwargs,
        }

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, window_size, input_size).

        Returns:
            Predictions of shape (batch_size, horizon).
        """
        ...

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        best_val_loss: float = float("inf"),
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a training checkpoint.

        Args:
            path: File path for the checkpoint.
            optimizer: Optimizer state to save.
            scheduler: Scheduler state to save.
            epoch: Current epoch number.
            best_val_loss: Best validation loss so far.
            extra: Additional metadata to include.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "hparams": self.hparams,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path} (epoch={epoch})")

    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load a training checkpoint.

        Robust to `module.` prefixes produced by DataParallel/DistributedDataParallel
        so checkpoints remain portable across CPU/GPU and single/multi-GPU setups.

        Args:
            path: Path to the checkpoint file.
            optimizer: Optimizer to restore state into.
            scheduler: Scheduler to restore state into.
            device: Device to map tensors to.

        Returns:
            Dictionary with checkpoint metadata (epoch, best_val_loss, etc.).

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = device if device else "cpu"
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        state_dict = checkpoint["model_state_dict"]

        # Try direct load first; if keys mismatch due to 'module.' prefix, try fixing it.
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            # strip 'module.' prefix if present
            stripped = {k.replace("module.", "") if k.startswith("module.") else k: v
                        for k, v in state_dict.items()}
            try:
                self.load_state_dict(stripped)
            except RuntimeError:
                # as a last resort, add 'module.' prefix to keys (for loading DP checkpoint into DP model)
                prefixed = {f"module.{k}": v for k, v in state_dict.items()}
                self.load_state_dict(prefixed)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            f"Checkpoint loaded: {path} (epoch={checkpoint.get('epoch', 0)})"
        )

        return {
            "epoch": checkpoint.get("epoch", 0),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "hparams": checkpoint.get("hparams", {}),
            "extra": checkpoint.get("extra", {}),
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model.

        Returns:
            Dictionary with model name, parameters, and hyperparameters.
        """
        return {
            "model_name": self.model_name,
            "n_parameters": self.count_parameters(),
            "hparams": self.hparams,
        }
