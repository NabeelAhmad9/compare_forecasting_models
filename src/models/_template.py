"""
Model Template for New Forecasting Models.

INSTRUCTIONS FOR ADDING A NEW MODEL:
=====================================

1. Create a new file in src/models/ (e.g., `mymodel.py`)
2. Copy this template and implement the required methods
3. Create configs/models/MyModel/search_space.yaml for HPO
4. The model will be auto-discovered - no other code changes needed!

REQUIREMENTS:
- Inherit from BaseForecaster
- Use @register_model("ModelName") decorator
- Implement forward() method
- Input shape: (batch, window_size, input_size)
- Output shape: (batch, horizon)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.base import BaseForecaster
from src.models import register_model


# Use the decorator with your model name (must match config folder name)
@register_model("ModelTemplate")
class ModelTemplateForecaster(BaseForecaster):
    """Template forecasting model.

    Replace this docstring with your model description.

    Args:
        input_size: Number of input features (default: 5 for OHLCV).
        window_size: Length of input sequence (lookback window).
        horizon: Number of future steps to predict.
        hidden_size: Example hyperparameter.
        num_layers: Example hyperparameter.
        dropout: Dropout rate.
        **kwargs: Additional hyperparameters (stored in self.hparams).
    """

    def __init__(
        self,
        input_size: int = 5,
        window_size: int = 24,
        horizon: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        # Always call super().__init__ with model_name and all hyperparameters
        super().__init__(
            model_name="ModelTemplate",  # Must match @register_model name
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Store hyperparameters as instance attributes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define your model architecture here
        # Example:
        self.encoder = nn.Linear(input_size * window_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, window_size, input_size).
               Contains the historical data for forecasting.

        Returns:
            Predictions of shape (batch_size, horizon).
            Direct multi-step forecast (NOT recursive).

        Note:
            - Validate input shape at the start
            - Validate output shape before returning
            - Use direct forecasting (predict all horizon steps at once)
        """
        # Validate input shape
        batch_size, seq_len, features = x.shape
        assert seq_len == self.window_size, (
            f"Expected window_size={self.window_size}, got {seq_len}"
        )
        assert features == self.input_size, (
            f"Expected input_size={self.input_size}, got {features}"
        )

        # Implement your forward pass here
        # Example: Flatten -> Encode -> Decode
        x = x.reshape(batch_size, -1)  # (batch, window_size * input_size)
        x = self.encoder(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.decoder(x)  # (batch, horizon)

        # Validate output shape
        assert output.shape == (batch_size, self.horizon), (
            f"Expected output shape ({batch_size}, {self.horizon}), got {output.shape}"
        )

        return output


# =============================================================================
# SEARCH SPACE TEMPLATE (save as configs/models/ModelTemplate/search_space.yaml)
# =============================================================================
"""
# configs/models/ModelTemplate/search_space.yaml

search_space:
  hidden_size:
    type: int
    low: 32
    high: 256
  
  num_layers:
    type: int
    low: 1
    high: 4
  
  dropout:
    type: float
    low: 0.0
    high: 0.5
    step: 0.05
  
  learning_rate:
    type: loguniform
    low: 0.0001
    high: 0.01
  
  batch_size:
    type: categorical
    choices: [32, 64, 128]
"""
