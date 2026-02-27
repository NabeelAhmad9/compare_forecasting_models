"""
N-HiTS Forecaster.

Neural Hierarchical Interpolation for Time Series forecasting.
Uses hierarchical interpolation and multi-rate signal sampling
via pooling to capture patterns at different frequencies.
Based on "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting".
Refactored to match official univariate architecture.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Tuple, Optional

from src.models.base import BaseForecaster
from src.models import register_model


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str = 'linear'):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, self.backcast_size:]

        # Interpolate forecast
        if self.interpolation_mode == 'nearest':
            forecast = F.interpolate(forecast.unsqueeze(1), size=self.forecast_size, mode='nearest').squeeze(1)
        elif self.interpolation_mode == 'linear':
            forecast = F.interpolate(forecast.unsqueeze(1), size=self.forecast_size, mode='linear', align_corners=False).squeeze(1)
        elif self.interpolation_mode == 'cubic':
            forecast = F.interpolate(forecast.unsqueeze(1), size=self.forecast_size, mode='cubic', align_corners=False).squeeze(1)
            
        return backcast, forecast


class NHiTSBlock(nn.Module):
    """N-HiTS block which produces basis coefficients and interpolates them."""
    
    def __init__(
        self,
        n_time_in: int,
        n_time_out: int,
        n_theta: int,
        n_hidden: int,
        n_layers: int,
        pool_kernel_size: int,
        pooling_mode: str,
        n_freq_downsample: int,
        dropout: float,
        activation: str,
        batch_normalization: bool,
        interpolation_mode: str,
    ) -> None:
        super().__init__()
        
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_theta = n_theta
        self.pool_kernel_size = pool_kernel_size
        self.pooling_mode = pooling_mode
        self.n_freq_downsample = n_freq_downsample
        
        # Pooling
        if pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, ceil_mode=True)
        elif pooling_mode == 'average':
            self.pooling_layer = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, ceil_mode=True)
        else:
            raise ValueError(f"Invalid pooling mode: {pooling_mode}")
            
        # MLP
        layers = []
        # Input size is pooled length
        # ceil(n_time_in / pool_kernel_size)
        input_size = int(np.ceil(n_time_in / pool_kernel_size))
        
        # Create MLP layers
        for i in range(n_layers):
            in_dim = input_size if i == 0 else n_hidden
            out_dim = n_hidden
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'Softplus':
                layers.append(nn.Softplus())
            elif activation == 'Tanh':
                layers.append(nn.Tanh())
            elif activation == 'SELU':
                layers.append(nn.SELU())
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            elif activation == 'PReLU':
                layers.append(nn.PReLU())
            elif activation == 'Sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU()) # Default
                
            if batch_normalization:
                layers.append(nn.BatchNorm1d(out_dim))
                
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
        # Final projection to theta
        layers.append(nn.Linear(n_hidden, n_theta))
        self.layers = nn.Sequential(*layers)
        
        # Basis
        self.basis = IdentityBasis(
            backcast_size=n_time_in,
            forecast_size=n_time_out,
            interpolation_mode=interpolation_mode
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, n_time_in)
        
        # Pooling expects (B, C, L) -> (B, 1, n_time_in)
        x_in = x.unsqueeze(1)
        x_pooled = self.pooling_layer(x_in) 
        x_pooled = x_pooled.squeeze(1) # (B, pooled_len)
        
        theta = self.layers(x_pooled)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


@register_model("N-HiTS")
class NHiTSForecaster(BaseForecaster):
    """N-HiTS forecasting model.

    Faithful implementation of the official N-HiTS architecture for univariate forecasting.
    
    Args:
        input_size: Number of input features (only used to select target, internal model is univariate).
        window_size: Input sequence length.
        horizon: Forecast horizon.
        n_stacks: Number of stacks.
        n_blocks: Number of blocks per stack.
        n_layers: Number of MLP layers per block.
        n_hidden: MLP hidden units.
        n_pool_kernel_size: List of pooling kernel sizes (one per stack).
        n_freq_downsample: List of downsampling factors (one per stack).
        pooling_mode: 'max' or 'average'.
        interpolation_mode: 'linear', 'nearest', 'cubic'.
        dropout: Dropout rate.
        activation: Activation function string.
        batch_normalization: Whether to use batch norm.
        shared_weights: Whether blocks in a stack share weights.
        target_idx: Index of target variable in multivariate input.
    """

    def __init__(
        self,
        input_size: int = 5,
        window_size: int = 168,
        horizon: int = 4,
        n_stacks: int = 3,
        n_blocks: int = 1,
        n_layers: int = 2,
        n_hidden: int = 512,
        n_pool_kernel_size: List[int] = None,
        n_freq_downsample: List[int] = None,
        pooling_mode: str = 'max',
        interpolation_mode: str = 'linear',
        dropout: float = 0.1,
        activation: str = 'ReLU',
        batch_normalization: bool = False,
        shared_weights: bool = False,
        target_idx: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name="N-HiTS",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            n_stacks=n_stacks,
            n_blocks=n_blocks,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_pool_kernel_size=n_pool_kernel_size or [2, 2, 2],
            n_freq_downsample=n_freq_downsample or [4, 2, 1],
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout=dropout,
            activation=activation,
            batch_normalization=batch_normalization,
            shared_weights=shared_weights,
            target_idx=target_idx,
            **kwargs,
        )

        self.target_idx = target_idx
        self.n_time_in = window_size
        self.n_time_out = horizon
        
        if n_pool_kernel_size is None:
            n_pool_kernel_size = [2, 2, 2]
        if n_freq_downsample is None:
            n_freq_downsample = [4, 2, 1]
            
        # Validate lists match n_stacks
        assert len(n_pool_kernel_size) == n_stacks, "n_pool_kernel_size must match n_stacks"
        assert len(n_freq_downsample) == n_stacks, "n_freq_downsample must match n_stacks"

        self.stacks = nn.ModuleList()
        
        for s in range(n_stacks):
            blocks = nn.ModuleList()
            
            # Calculate n_theta
            # n_theta = n_time_in + max(n_time_out // n_freq_downsample[s], 1)
            n_theta = self.n_time_in + max(self.n_time_out // n_freq_downsample[s], 1)
            
            if shared_weights:
                # If shared weights, create one block and reuse it
                block = NHiTSBlock(
                    n_time_in=self.n_time_in,
                    n_time_out=self.n_time_out,
                    n_theta=n_theta,
                    n_hidden=n_hidden,
                    n_layers=n_layers,
                    pool_kernel_size=n_pool_kernel_size[s],
                    pooling_mode=pooling_mode,
                    n_freq_downsample=n_freq_downsample[s],
                    dropout=dropout,
                    activation=activation,
                    batch_normalization=batch_normalization,
                    interpolation_mode=interpolation_mode
                )
                for _ in range(n_blocks):
                    blocks.append(block)
            else:
                for _ in range(n_blocks):
                    blocks.append(
                        NHiTSBlock(
                            n_time_in=self.n_time_in,
                            n_time_out=self.n_time_out,
                            n_theta=n_theta,
                            n_hidden=n_hidden,
                            n_layers=n_layers,
                            pool_kernel_size=n_pool_kernel_size[s],
                            pooling_mode=pooling_mode,
                            n_freq_downsample=n_freq_downsample[s],
                            dropout=dropout,
                            activation=activation,
                            batch_normalization=batch_normalization,
                            interpolation_mode=interpolation_mode
                        )
                    )
            
            self.stacks.append(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (batch_size, window_size, input_size).

        Returns:
            Predictions (batch_size, horizon).
        """
        # Extract target variable only
        # x is (B, T, C)
        insample_y = x[:, :, self.target_idx] # (B, n_time_in)
        
        # Standard residual mechanism (no flip, no naive level init)
        residuals = insample_y  # (B, n_time_in)
        forecast = torch.zeros(
            insample_y.shape[0], self.n_time_out, device=insample_y.device
        )
        
        # Hierarchical interpolation through stacks and blocks
        for blocks in self.stacks:
            for block in blocks:
                backcast, block_forecast = block(residuals)
                residuals = residuals - backcast
                forecast = forecast + block_forecast
                
        return forecast
