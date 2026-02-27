"""
Autoformer Forecaster.

Standard Autoformer architecture adapted for the research benchmarking framework.
Based on "Autoformer: Decomposition Transformers with Auto-Correlation for 
Long-Term Series Forecasting".
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster
from src.models import register_model

# Relative imports from current package
from .layers.Embed import DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
)

@register_model("Autoformer")
class AutoformerForecaster(BaseForecaster):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity.
    """

    def __init__(
        self,
        input_size: int = 5,
        window_size: int = 96,
        horizon: int = 4,
        moving_avg: int = 25,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 256,
        factor: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        embed: str = "timeF",
        freq: str = "h",
        output_attention: bool = False,
        target_idx: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Autoformer.

        Args:
            input_size: Number of input features (OHLCV = 5).
            window_size: Lookback window length.
            horizon: Prediction horizon.
            moving_avg: Kernel size for series decomposition.
            d_model: Hidden dimension.
            n_heads: Number of attention heads.
            e_layers: Number of encoder layers.
            d_layers: Number of decoder layers.
            d_ff: Feed-forward network dimension.
            factor: Attention factor.
            dropout: Dropout rate.
            activation: Activation function ('gelu' or 'relu').
            embed: Embedding type.
            freq: Frequency for temporal embedding.
            output_attention: Whether to output attention maps.
            target_idx: Index of the target feature to return (default: 3).
        """
        super().__init__(
            model_name="Autoformer",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            moving_avg=moving_avg,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            activation=activation,
            embed=embed,
            freq=freq,
            output_attention=output_attention,
            target_idx=target_idx,
            **kwargs,
        )

        self.seq_len = window_size
        self.label_len = window_size // 2
        self.pred_len = horizon
        self.output_attention = output_attention
        self.target_idx = target_idx

        # Decomp
        self.decomp = series_decomp(moving_avg)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            input_size, d_model, embed, freq, dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            input_size, d_model, embed, freq, dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True, factor, attention_dropout=dropout, output_attention=False
                        ),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False, factor, attention_dropout=dropout, output_attention=False
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    input_size,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, input_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Autoformer.

        Args:
            x: Input tensor of shape (batch, window_size, input_size)

        Returns:
            Predictions of shape (batch, horizon)
        """
        # Validate input shape
        batch_size, seq_len, features = x.shape
        assert (
            seq_len == self.window_size
        ), f"Expected window_size {self.window_size}, got {seq_len}"
        assert (
            features == self.input_size
        ), f"Expected input_size {self.input_size}, got {features}"

        # Ensure model and input are on the same device and correct dtype
        if any(True for _ in self.parameters()):
            model_device = next(self.parameters()).device
            assert (
                x.device == model_device
            ), f"Input device ({x.device}) must match model device ({model_device}). Call model.to(device) or move input to the model device."
        assert x.dtype.is_floating_point, f"Input tensor must be a floating dtype, got {x.dtype}"

        # Create dummy marks (Autoformer handles them in DataEmbedding_wo_pos)
        # embed='timeF' uses 4 features for 'h' — make marks device/dtype-safe
        device = x.device
        dtype = x.dtype
        x_mark_enc = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
        x_mark_dec = torch.zeros(
            batch_size, self.label_len + self.pred_len, 4, device=device, dtype=dtype
        )

        # decomp init (device-preserving)
        mean = torch.mean(x, dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        zeros = torch.zeros([batch_size, self.pred_len, features], device=device, dtype=dtype)
        seasonal_init, trend_init = self.decomp(x)

        # decoder input
        trend_init_combined = torch.cat(
            [trend_init[:, -self.label_len :, :], mean], dim=1
        )
        seasonal_init_combined = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )

        # enc
        enc_out = self.enc_embedding(x, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # dec
        dec_out = self.dec_embedding(seasonal_init_combined, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init_combined
        )
        # final
        dec_out = trend_part + seasonal_part

        # Extract last pred_len steps and the target feature
        output = dec_out[:, -self.pred_len :, self.target_idx]

        # Validate output shape
        assert output.shape == (
            batch_size,
            self.horizon,
        ), f"Expected output shape ({batch_size}, {self.horizon}), got {output.shape}"

        # Check for NaNs
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)

        return output
