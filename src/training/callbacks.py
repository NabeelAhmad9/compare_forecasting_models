"""
Training callbacks.

Implements early stopping and learning rate scheduling callbacks
for the training loop.
"""

from __future__ import annotations

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving.

    Attributes:
        patience: Number of epochs with no improvement to wait.
        min_delta: Minimum change to qualify as an improvement.
        counter: Current number of epochs without improvement.
        best_loss: Best validation loss observed.
        should_stop: Whether training should be stopped.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        verbose: bool = True,
    ) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum improvement threshold.
            verbose: Whether to log messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.counter}/{self.patience} "
                    f"(best={self.best_loss:.6f}, current={val_loss:.6f})"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info("EarlyStopping triggered!")
                return True

        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
