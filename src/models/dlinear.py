"""
DLinear Forecaster (paper-faithful).

Exact, minimal DLinear implementation adapted for OHLCV financial series.
- Series decomposition: x = trend + seasonal (moving average)
- Two linear layers along time (L -> H): seasonal and trend
- Supports `individual` (per-channel weights) or shared weights
- Predicts the `close` channel only (univariate output)

Notes:
- No hidden MLPs, no activations, no extra projections beyond selecting Close
- Weight initialization is Xavier for stability
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.base import BaseForecaster
from src.models import register_model


class MovingAvg(nn.Module):
    """Simple moving-average smoothing used for decomposition.

    Pads asymmetrically for even kernels to preserve sequence length.
    """

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        out = self.avg(x_padded.permute(0, 2, 1))  # (B, C, L)
        return out.permute(0, 2, 1)  # (B, L, C)


class SeriesDecomp(nn.Module):
    """Decomposition module returning (seasonal, trend)."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (seasonal=residual, trend=moving_mean)."""
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


@register_model("DLinear")
class DLinearForecaster(BaseForecaster):
    """Paper-accurate DLinear.

    Args:
        kernel_size: moving-average kernel for decomposition
        individual: if True, separate Linear per channel (L->H)
        target_idx: index of `close` in OHLCV (default 3)
    """

    def __init__(
        self,
        input_size: int = 5,
        window_size: int = 96,
        horizon: int = 24,
        individual: bool = False,
        kernel_size: int = 25,
        target_idx: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name="DLinear",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            individual=individual,
            kernel_size=kernel_size,
            target_idx=target_idx,
            **kwargs,
        )

        # Framework / dataset constraints (enforced strictly)
        assert input_size == 5, f"Expected 5 OHLCV features, got {input_size}"
        assert window_size in (24, 96), "Window size must be 24 or 96"
        assert horizon in (4, 24), "Horizon must be 4 or 24"
        assert 0 <= target_idx < input_size, "target_idx out of range"

        self.individual = bool(individual)
        self.kernel_size = int(kernel_size)
        self.target_idx = int(target_idx)

        # Decomposition
        self.decomposition = SeriesDecomp(self.kernel_size)

        # Linear heads (time-dim: L -> H). No extra layers allowed.
        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(window_size, horizon, bias=True) for _ in range(input_size)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(window_size, horizon, bias=True) for _ in range(input_size)]
            )
        else:
            # Shared mapping applied to every channel (callable on (B, C, L))
            self.linear_seasonal = nn.Linear(window_size, horizon, bias=True)
            self.linear_trend = nn.Linear(window_size, horizon, bias=True)

        # Stable initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for linear weights, zero biases."""
        def _init(layer: nn.Linear) -> None:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        if self.individual:
            for l in list(self.linear_seasonal) + list(self.linear_trend):
                _init(l)
        else:
            _init(self.linear_seasonal)
            _init(self.linear_trend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, L, 5) where L in {24,96}
        Returns:
            y: (B, H) — predictions for Close only (paper: direct multi-step)
        """
        assert x.ndim == 3, "Expected input shape (B, L, C)"
        B, L, C = x.shape
        assert L == self.window_size and C == self.input_size
        assert torch.isfinite(x).all(), "Input contains NaN/Inf"

        # Decompose into seasonal (residual) and trend (moving mean)
        seasonal, trend = self.decomposition(x)  # both (B, L, C)

        # Map L -> H (apply linear along last dim after appropriate permute)
        if self.individual:
            # list comprehension + stack -> (B, H, C)
            seasonal_out = torch.stack(
                [self.linear_seasonal[i](seasonal[:, :, i]) for i in range(C)], dim=-1
            )
            trend_out = torch.stack(
                [self.linear_trend[i](trend[:, :, i]) for i in range(C)], dim=-1
            )
        else:
            # seasonal.permute(0,2,1): (B, C, L) -> Linear -> (B, C, H)
            seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            # now (B, H, C)

        out_multivariate = seasonal_out + trend_out  # (B, H, C)

        # Select Close channel as final univariate forecast
        out_close = out_multivariate[:, :, self.target_idx]  # (B, H)

        assert out_close.shape == (B, self.horizon)
        assert torch.isfinite(out_close).all(), "Output contains NaN/Inf"

        return out_close
