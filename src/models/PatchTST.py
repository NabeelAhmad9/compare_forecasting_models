"""
PatchTST Forecaster.

Patching Time Series Transformer that segments input sequences into patches
and applies self-attention across the patch tokens. Supports series decomposition,
RevIN normalization, and individual channel modeling. Based on "A Time Series is
Worth 64 Words: Long-term Forecasting with Transformers".
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseForecaster
from src.models import register_model
from src.models.layers.PatchTST_backbone import PatchTST_backbone
from src.models.layers.PatchTST_layers import series_decomp


@register_model("PatchTST")
class PatchTST(BaseForecaster):
    """PatchTST forecasting model.

    Applies patching and self-attention to time series sequences. Optionally
    decomposes the series into trend and residual components, applying separate
    transformer backbones to each. Supports RevIN normalization and individual
    linear heads per feature.

    Args:
        input_size: Number of input features (e.g., 5 for OHLCV).
        window_size: Length of the input sequence.
        horizon: Number of future steps to forecast.
        patch_len: Length of each patch.
        stride: Stride for patching (overlap = patch_len - stride).
        d_model: Transformer model dimension.
        n_heads: Number of attention heads.
        e_layers: Number of encoder layers.
        d_ff: Feedforward network dimension.
        dropout: Dropout rate.
        fc_dropout: Dropout rate in fully connected layers.
        head_dropout: Dropout rate in projection head.
        individual: If True, use separate linear layers per feature.
        padding_patch: Padding strategy ('end' or None).
        revin: Whether to apply RevIN normalization.
        affine: Whether to use affine transformation in RevIN.
        subtract_last: Whether to subtract last value in RevIN.
        decomposition: Whether to decompose into trend and residual.
        kernel_size: Moving average kernel size for decomposition.
        target_idx: Index of the target variable to forecast (default 3 for Close).
        max_seq_len: Maximum sequence length for positional encoding.
        d_k: Key dimension (default d_model // n_heads).
        d_v: Value dimension (default d_model // n_heads).
        norm: Normalization type ('BatchNorm' or 'LayerNorm').
        attn_dropout: Dropout rate in attention.
        act: Activation function ('gelu' or 'relu').
        key_padding_mask: Whether to use key padding mask.
        padding_var: Padding variance (if applicable).
        attn_mask: Attention mask tensor.
        res_attention: Whether to use residual attention.
        pre_norm: Whether to apply pre-normalization.
        store_attn: Whether to store attention weights.
        pe: Positional encoding type ('zeros', 'normal', etc.).
        learn_pe: Whether to learn positional encodings.
        pretrain_head: Whether to use pretraining head.
        head_type: Type of projection head ('flatten').
        verbose: Whether to print detailed info.
        **kwargs: Additional hyperparameters.
    """

    def __init__(
        self,
        input_size: int = 5,
        window_size: int = 168,
        horizon: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        individual: bool = False,
        padding_patch: Optional[str] = None,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        decomposition: bool = False,
        kernel_size: int = 25,
        target_idx: int = 3,
        max_seq_len: int = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = 'flatten',
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name="PatchTST",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            individual=individual,
            padding_patch=padding_patch,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            decomposition=decomposition,
            kernel_size=kernel_size,
            target_idx=target_idx,
            **kwargs,
        )

        self.target_idx = target_idx
        self.decomposition = decomposition
        
        # Map framework parameters to backbone parameters
        c_in = input_size
        context_window = window_size
        target_window = horizon
        n_layers = e_layers
        
        # Build backbone model(s)
        # The original implementation constructs either:
        # - Two separate backbones (trend + residual) if decomposition=True
        # - One backbone if decomposition=False
        # We preserve this architecture exactly as implemented
        
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=c_in, 
                context_window=context_window, 
                target_window=target_window, 
                patch_len=patch_len, 
                stride=stride,
                max_seq_len=max_seq_len, 
                n_layers=n_layers, 
                d_model=d_model,
                n_heads=n_heads, 
                d_k=d_k, 
                d_v=d_v, 
                d_ff=d_ff, 
                norm=norm, 
                attn_dropout=attn_dropout,
                dropout=dropout, 
                act=act, 
                key_padding_mask=key_padding_mask, 
                padding_var=padding_var,
                attn_mask=attn_mask, 
                res_attention=res_attention, 
                pre_norm=pre_norm, 
                store_attn=store_attn,
                pe=pe, 
                learn_pe=learn_pe, 
                fc_dropout=fc_dropout, 
                head_dropout=head_dropout, 
                padding_patch=padding_patch,
                pretrain_head=pretrain_head, 
                head_type=head_type, 
                individual=individual, 
                revin=revin, 
                affine=affine,
                subtract_last=subtract_last, 
                verbose=verbose, 
                **kwargs
            )
            self.model_res = PatchTST_backbone(
                c_in=c_in, 
                context_window=context_window, 
                target_window=target_window, 
                patch_len=patch_len, 
                stride=stride,
                max_seq_len=max_seq_len, 
                n_layers=n_layers, 
                d_model=d_model,
                n_heads=n_heads, 
                d_k=d_k, 
                d_v=d_v, 
                d_ff=d_ff, 
                norm=norm, 
                attn_dropout=attn_dropout,
                dropout=dropout, 
                act=act, 
                key_padding_mask=key_padding_mask, 
                padding_var=padding_var,
                attn_mask=attn_mask, 
                res_attention=res_attention, 
                pre_norm=pre_norm, 
                store_attn=store_attn,
                pe=pe, 
                learn_pe=learn_pe, 
                fc_dropout=fc_dropout, 
                head_dropout=head_dropout, 
                padding_patch=padding_patch,
                pretrain_head=pretrain_head, 
                head_type=head_type, 
                individual=individual, 
                revin=revin, 
                affine=affine,
                subtract_last=subtract_last, 
                verbose=verbose, 
                **kwargs
            )
        else:
            self.model = PatchTST_backbone(
                c_in=c_in, 
                context_window=context_window, 
                target_window=target_window, 
                patch_len=patch_len, 
                stride=stride,
                max_seq_len=max_seq_len, 
                n_layers=n_layers, 
                d_model=d_model,
                n_heads=n_heads, 
                d_k=d_k, 
                d_v=d_v, 
                d_ff=d_ff, 
                norm=norm, 
                attn_dropout=attn_dropout,
                dropout=dropout, 
                act=act, 
                key_padding_mask=key_padding_mask, 
                padding_var=padding_var,
                attn_mask=attn_mask, 
                res_attention=res_attention, 
                pre_norm=pre_norm, 
                store_attn=store_attn,
                pe=pe, 
                learn_pe=learn_pe, 
                fc_dropout=fc_dropout, 
                head_dropout=head_dropout, 
                padding_patch=padding_patch,
                pretrain_head=pretrain_head, 
                head_type=head_type, 
                individual=individual, 
                revin=revin, 
                affine=affine,
                subtract_last=subtract_last, 
                verbose=verbose, 
                **kwargs
            )
    
    def forward(
        self, x: Tensor, x_mark_enc: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with strict shape validation.

        Args:
            x: Input tensor of shape (batch_size, window_size, input_size).
            x_mark_enc: Optional time markers (not used, included for interface compatibility).

        Returns:
            Predictions of shape (batch_size, horizon) for the target variable.

        Shapes:
            - Input x: (B, window_size, num_features)
            - Output: (B, horizon)
        """
        # Strict input validation
        assert x.ndim == 3, (
            f"Expected 3D input (batch, seq_len, features), got {x.ndim}D tensor"
        )
        assert x.shape[1] == self.window_size, (
            f"Expected sequence length {self.window_size}, got {x.shape[1]}"
        )
        assert x.shape[2] == self.input_size, (
            f"Expected {self.input_size} features, got {x.shape[2]}"
        )
        
        # Ensure finite inputs (detect NaN/Inf early)
        assert torch.isfinite(x).all(), "Input contains NaN or Inf values"
        
        # Original PatchTST expects input as (B, L, C) where:
        # - B is batch size
        # - L is sequence length
        # - C is number of channels/features
        # This matches our framework's convention, so no initial permutation needed
        
        # The backbone internally:
        # 1. Permutes to (B, C, L) for channel-wise processing
        # 2. Returns (B, C, H) where H is horizon
        # 3. Original code permutes back to (B, H, C)
        
        # Preserve exact tensor permutations from original implementation
        if self.decomposition:
            # Series decomposition path
            # decomp_module expects (B, L, C) and returns (residual, trend) both (B, L, C)
            res_init, trend_init = self.decomp_module(x)
            
            # Permute to (B, C, L) for backbone processing
            res_init = res_init.permute(0, 2, 1)  # (B, C, L)
            trend_init = trend_init.permute(0, 2, 1)  # (B, C, L)
            
            # Process through separate backbones
            res = self.model_res(res_init)  # (B, C, H)
            trend = self.model_trend(trend_init)  # (B, C, H)
            
            # Combine predictions
            x_out = res + trend  # (B, C, H)
            
            # Permute to (B, H, C)
            x_out = x_out.permute(0, 2, 1)  # (B, H, C)
        else:
            # Standard path without decomposition
            # Permute to (B, C, L) for backbone
            x_permuted = x.permute(0, 2, 1)  # (B, C, L)
            
            # Process through backbone
            x_out = self.model(x_permuted)  # (B, C, H)
            
            # Permute to (B, H, C)
            x_out = x_out.permute(0, 2, 1)  # (B, H, C)
        
        # Extract target variable prediction for univariate forecasting
        # x_out shape: (B, H, C)
        # We select the target channel to get (B, H)
        output = x_out[:, :, self.target_idx]  # (B, H)
        
        # Validate output shape
        assert output.shape == (x.shape[0], self.horizon), (
            f"Expected output shape ({x.shape[0]}, {self.horizon}), "
            f"got {output.shape}"
        )
        
        # Ensure finite outputs
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        return output