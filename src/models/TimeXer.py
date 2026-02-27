"""
TimeXer Forecaster.

Implementation of TimeXer, which utilizes global tokens and cross-attention
between patched target features and other exogenous variables for effective
time series forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

from src.models.base import BaseForecaster
from src.models import register_model
from src.models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from src.models.layers.Embed import DataEmbedding_inverted, PositionalEmbedding


class FlattenHead(nn.Module):
    """Linear head for classification/regression tasks."""

    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_vars, d_model, patch_num).

        Returns:
            Output tensor of shape (batch, n_vars, target_window).
        """
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    """Embedding for patched variables with a global token."""

    def __init__(self, n_vars: int, d_model: int, patch_len: int, dropout: float) -> None:
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Apply patching and embedding.

        Args:
            x: Input tensor (batch, n_vars, seq_len).

        Returns:
            Tuple of (embedded tensor, number of variables).
        """
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        # Shape: (B, N, patch_num, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        
        # Reshape for embedding: (B*N, patch_num, patch_len)
        batch_size = x.shape[0]
        patch_num = x.shape[2]
        x = torch.reshape(x, (batch_size * n_vars, patch_num, self.patch_len))
        
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        
        # Reshape back and add global token
        x = torch.reshape(x, (batch_size, n_vars, patch_num, -1))
        x = torch.cat([x, glb], dim=2)  # (B, N, patch_num + 1, d_model)
        
        # Final combine: (B*N, patch_num + 1, d_model)
        x = torch.reshape(x, (batch_size * n_vars, patch_num + 1, -1))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, layers: list[EncoderLayer], norm_layer: Optional[nn.Module] = None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(
        self, 
        x: torch.Tensor, 
        cross: torch.Tensor, 
        x_mask: Optional[torch.Tensor] = None, 
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder layers.

        Args:
            x: Primary input (patched).
            cross: Cross-attention input (inverted variates).
            x_mask: Attention mask for primary input.
            cross_mask: Attention mask for cross input.

        Returns:
            Encoded features.
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    """Single layer of TimeXer encoder with cross-attention."""

    def __init__(
        self, 
        self_attention: AttentionLayer, 
        cross_attention: AttentionLayer, 
        d_model: int, 
        d_ff: Optional[int] = None,
        dropout: float = 0.1, 
        activation: str = "relu"
    ) -> None:
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self, 
        x: torch.Tensor, 
        cross: torch.Tensor, 
        x_mask: Optional[torch.Tensor] = None, 
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B*N, patch_num+1, d_model).
            cross: Cross features (B, N_other, d_model).
            x_mask: Mask for self-attention.
            cross_mask: Mask for cross-attention.

        Returns:
            Updated features.
        """
        b_n, _, d = x.shape
        b, n_other, _ = cross.shape
        
        # Self-attention
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        # Cross-attention via global token (the last token in each patched sequence)
        x_glb_ori = x[:, -1:, :]  # (B*N, 1, D)
        
        # Reshape global tokens to participate in cross-attention per-batch
        # x_glb: (B, N_vars, D)
        n_vars = b_n // b
        x_glb = torch.reshape(x_glb_ori, (b, n_vars, d))
        
        # Cross-attention: queries=global tokens, keys/values=inverted variates
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross, attn_mask=cross_mask
        )[0])
        
        # Reshape back to (B*N, 1, D)
        x_glb_attn = torch.reshape(x_glb_attn, (b_n, 1, d))
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        # Combine updated global token with patches
        x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # FFN
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


@register_model("TimeXer")
class TimeXerForecaster(BaseForecaster):
    """TimeXer forecasting model.

    Optimized implementation for research auditing, supporting absolute imports
    and standardized API.

    Args:
        input_size: Number of input features (default 5 for OHLCV).
        window_size: Lookback window size.
        horizon: Forecast horizon.
        patch_len: Length of each patch.
        d_model: Dimension of the model.
        d_ff: Dimension of the FFN.
        e_layers: Number of encoder layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        factor: Attention factor.
        activation: Activation function ('relu' or 'gelu').
        use_norm: Whether to use non-stationary normalization.
        target_idx: Index of the target feature (default 3 for 'close').
    """

    def __init__(
        self,
        input_size: int = 5,
        window_size: int = 168,
        horizon: int = 4,
        patch_len: int = 16,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        factor: int = 5,
        activation: str = "gelu",
        use_norm: bool = True,
        target_idx: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name="TimeXer",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            patch_len=patch_len,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            n_heads=n_heads,
            dropout=dropout,
            factor=factor,
            activation=activation,
            use_norm=use_norm,
            target_idx=target_idx,
            **kwargs,
        )

        self.patch_len = patch_len
        self.use_norm = use_norm
        self.target_idx = target_idx
        
        if window_size % patch_len != 0:
            raise ValueError(f"window_size ({window_size}) must be divisible by patch_len ({patch_len})")
            
        self.patch_num = window_size // patch_len
        
        # Univariate patching for the target feature
        self.en_embedding = EnEmbedding(1, d_model, patch_len, dropout)

        # Inverted embedding for exogenous features
        self.ex_embedding = DataEmbedding_inverted(window_size, d_model, 'fixed', 'h', dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(1, self.head_nf, horizon, head_dropout=dropout)

    def forward(self, x: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (batch, window_size, input_size).
            x_mark_enc: Optional time markers.

        Returns:
            Forecast (batch, horizon).
        """
        # 1. Validation
        assert x.ndim == 3, f"Expected 3D input (B, T, N), got {x.ndim}D"
        assert x.shape[1] == self.window_size, f"Expected window_size {self.window_size}, got {x.shape[1]}"
        assert x.shape[2] == self.input_size, f"Expected input_size {self.input_size}, got {x.shape[2]}"

        # 2. Normalization
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # 3. Feature Selection
        # Target feature: (B, T, 1)
        target = x[:, :, self.target_idx : self.target_idx + 1]
        
        # Exogenous features: All features except target
        mask = torch.ones(x.shape[2], dtype=torch.bool, device=x.device)
        mask[self.target_idx] = False
        exogenous = x[:, :, mask]

        # 4. Embeddings
        # Patching target: (B, 1, T) -> (B*1, patches+1, d_model)
        en_embed, _ = self.en_embedding(target.permute(0, 2, 1))

        # Inverted exogenous: (B, N-1, T) -> (B, N-1, d_model)
        ex_embed = self.ex_embedding(exogenous.permute(0, 2, 1), x_mark_enc)

        # 5. Encoder
        enc_out = self.encoder(en_embed, ex_embed)
        
        # 6. Prediction Head
        # Reshape enc_out: (B, 1, patches+1, d_model)
        enc_out = torch.reshape(enc_out, (x.shape[0], 1, self.patch_num + 1, -1))
        # Head expects: (B, N, d_model, patches+1)
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)  # (B, 1, horizon)
        dec_out = dec_out.squeeze(1)  # (B, horizon)

        # 7. De-Normalization
        if self.use_norm:
            target_stdev = stdev[:, 0, self.target_idx].unsqueeze(1)
            target_mean = means[:, 0, self.target_idx].unsqueeze(1)
            dec_out = dec_out * target_stdev + target_mean

        # 8. Check for NaNs
        if torch.isnan(dec_out).any():
            dec_out = torch.nan_to_num(dec_out)

        assert dec_out.shape == (x.shape[0], self.horizon), f"Expected {(x.shape[0], self.horizon)}, got {dec_out.shape}"
        return dec_out
