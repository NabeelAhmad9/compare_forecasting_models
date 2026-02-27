"""
iTransformer Forecaster.

Inverted Transformer that applies attention across the variate (feature)
dimension rather than the temporal dimension, treating each feature as
a token. Based on "iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting".
"""

from __future__ import annotations

import math
from typing import Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster
from src.models.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from src.models.layers.Embed import DataEmbedding_inverted
from src.models import register_model

@register_model("iTransformer")
class iTransformerForecaster(BaseForecaster):
    """iTransformer forecasting model.

    Optimized implementation matching official architecture with
    non-stationary normalization and inverted embedding.

    Args:
        input_size: Number of input features (variates).
        window_size: Input sequence length.
        horizon: Forecast horizon.
        d_model: Model dimension (embedding of each variate's time series).
        d_ff: FFN inner dimension.
        e_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        activation: Activation function ('relu' or 'gelu').
        target_idx: Index of the target variable to forecast (default 3).
    """

    def __init__(
        self,
        input_size: int = 5,
        window_size: int = 24,
        horizon: int = 4,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        target_idx: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name="iTransformer",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            target_idx=target_idx,
            **kwargs,
        )

        self.target_idx = target_idx
        # Inverted embedding: maps window_size (time) to d_model
        self.enc_embedding = DataEmbedding_inverted(window_size, d_model, 'fixed', 'h', dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False, factor=5, attention_dropout=dropout, output_attention=False
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Project each variate embedding to horizon
        self.projector = nn.Linear(d_model, horizon, bias=True)

    def forward(
        self, x: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (batch_size, window_size, input_size).
            x_mark_enc: Optional time markers (not used currently).

        Returns:
            Predictions (batch_size, horizon).
        """
        # Normalization (Non-stationary Transformer)
        # x: (B, T, N)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x /= stdev

        # Inverted Transformer Logic
        # Permute to (B, N, T) so 'T' is the feature dimension for embedding
        x = x.permute(0, 2, 1)  # (B, N, T)

        # Embedding
        enc_out = self.enc_embedding(x, None)  # (B, N, d_model)

        # Encoder (Attention across variates N)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # (B, N, d_model)

        # Projection (B, N, d_model) -> (B, N, horizon)
        dec_out = self.projector(enc_out)

        # De-Normalization only for the target variate
        # dec_out is (B, N, horizon).
        # We perform univariate forecasting on the target channel
        
        target_pred = dec_out[:, self.target_idx, :]  # (B, horizon)
        
        # Get statistics for the target variate
        # stdev is (B, 1, N), means is (B, 1, N)
        target_stdev = stdev[:, 0, self.target_idx].unsqueeze(1) # (B, 1)
        target_mean = means[:, 0, self.target_idx].unsqueeze(1)  # (B, 1)

        # Denormalize
        output = target_pred * target_stdev + target_mean

        return output
