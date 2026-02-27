"""
LSTM + Dense projection forecaster (LSTM).

Architecture (strict):
 - 2-layer LSTM encoder (default num_layers = 2, configurable, batch_first=True)
 - use only the last time-step output
 - projection head: Dropout -> Linear(hidden*dirs -> mlp_hidden_size) -> ReLU -> Dropout -> Linear(mlp_hidden_size -> horizon)

Implementation follows repository conventions: inherits from BaseForecaster,
registered with @register_model("LSTM"), stores hyperparameters in
super().__init__ and validates tensor shapes/devices at entry points.

Constraints satisfied: no normalization, no seq-to-seq decoding, no time pooling.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.base import BaseForecaster
from src.models import register_model


@register_model("LSTM")
class LSTMForecaster(BaseForecaster):
    """LSTM encoder with a small MLP projection head.

    Args:
        input_size: Number of input features.
        window_size: Input sequence length (lookback).
        horizon: Forecast horizon (output length).
        hidden_size: LSTM hidden dimension.
        mlp_hidden_size: Hidden units in projection MLP.
        dropout: Dropout probability (between LSTM layers and in head).
        bidirectional: If True, use bidirectional LSTM.

    Notes:
        - num_layers: Number of stacked LSTM layers (default=2, configurable).
        - forward() validates input shape and device compatibility.
    """

    def __init__(
        self,
        input_size: int,
        window_size: int,
        horizon: int,
        hidden_size: int = 64,
        mlp_hidden_size: int = 128,
        dropout: float = 0.1,
        bidirectional: bool = False,
        num_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        # Persist hyperparameters via BaseForecaster
        super().__init__(
            model_name="LSTM",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            num_layers=num_layers,
            **kwargs,
        )

        # Architecture parameters (configurable)
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.mlp_hidden_size = int(mlp_hidden_size)
        self.dropout_p = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        # LSTM encoder (dropout only applied between layers)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        # Projection head uses last time-step only
        self.proj = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_size * self.num_directions, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.mlp_hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size, input_size).

        Returns:
            Tensor of shape (batch, horizon).
        """
        # Basic shape checks (fail fast)
        assert x.ndim == 3, f"Expected 3D input (B, W, F), got {x.ndim}D"
        batch, seq_len, feat = x.shape
        assert seq_len == self.window_size, (
            f"Expected window_size={self.window_size}, got {seq_len}"
        )
        assert feat == self.input_size, (
            f"Expected input_size={self.input_size}, got {feat}"
        )

        # Ensure device compatibility
        model_device = next(self.parameters()).device
        assert x.device == model_device, (
            f"Input device ({x.device}) does not match model device ({model_device})"
        )

        # LSTM encoder
        # lstm_out shape: (batch, seq_len, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)

        # Use only the last time-step output
        # last_output shape: (batch, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]

        # Projection head -> (batch, horizon)
        out = self.proj(last_output)

        # Final shape validation
        assert out.shape == (batch, self.horizon), (
            f"Expected output shape ({batch}, {self.horizon}), got {tuple(out.shape)}"
        )

        return out
