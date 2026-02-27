"""
TimesNet Forecaster.

Transforms 1D time series into 2D tensors via FFT-based period detection,
then applies 2D convolutions (Inception blocks) to capture both intra- and
inter-period variations. Based on "TimesNet: Temporal 2D-Variation Modeling".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from typing import Any, Optional, List

from src.models.base import BaseForecaster
from src.models import register_model


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        # Normalize period weights with softmax (official implementation)
        period_weight = F.softmax(period_weight, dim=1)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding (use actual input length T instead of seq_len + pred_len)
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - T), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from [B, d_model, n_periods, period]
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            
            res.append(out[:, :T, :] * period_weight[:, i].unsqueeze(1).unsqueeze(1).repeat(1, T, 1))
        
        res = torch.sum(torch.stack(res), dim=0)
        # Residual inside block
        return res + x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        return self.dropout(x)


@register_model("TimesNet")
class TimesNetForecaster(BaseForecaster):
    """TimesNet forecasting model.

    Faithful implementation of TimesNet architecture with Non-Stationary normalization
    and 2D structural variation modeling.
    
    Modified for fair HPO: Replaced direct temporal expansion with parameter-efficient
    prediction head to prevent parameter explosion across different window sizes.
    """

    def __init__(
        self,
        input_size: int = 5,
        window_size: int = 168,
        horizon: int = 4,
        d_model: int = 64,
        d_ff: int = 128,
        e_layers: int = 2,
        num_kernels: int = 6,
        top_k: int = 5,
        dropout: float = 0.1,
        target_idx: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name="TimesNet",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            num_kernels=num_kernels,
            top_k=top_k,
            dropout=dropout,
            target_idx=target_idx,
            **kwargs,
        )
        self.seq_len = window_size
        self.pred_len = horizon
        self.target_idx = target_idx

        # Embedding
        self.enc_embedding = DataEmbedding(input_size, d_model, dropout)

        # TimesBlocks (process at original sequence length)
        self.model = nn.ModuleList([
            TimesBlock(self.seq_len, self.pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(e_layers)
        ])
        
        # Layer Norms
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(e_layers)])

        # Parameter-efficient prediction head (replaces temporal expansion)
        # Maps from (B, T, d_model) to (B, H, C)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_size * horizon)
        )

    def forward(self, x: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (batch_size, window_size, input_size).

        Returns:
            Predictions (batch_size, horizon).
        """
        B = x.shape[0]
        
        # Normalization (Non-Stationary Transformer)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # Embedding
        enc_out = self.enc_embedding(x, x_mark_enc)  # [B, T, d_model]
        
        # TimesNet Blocks (no temporal expansion needed)
        for i in range(len(self.model)):
            # Note: TimesBlock expects (seq_len + pred_len) internally for padding
            # but returns same length as input after slicing
            enc_out = self.layer_norm[i](self.model[i](enc_out))
            
        # Use mean pooling over time to get fixed-size representation
        enc_pooled = enc_out.mean(dim=1)  # [B, d_model]
        
        # Prediction head: [B, d_model] -> [B, input_size * horizon]
        dec_out = self.prediction_head(enc_pooled)  # [B, input_size * horizon]
        
        # Reshape to [B, horizon, input_size]
        dec_out = dec_out.view(B, self.pred_len, self.input_size)
        
        # De-Normalization
        dec_out = dec_out * stdev + means
        
        # Select target variable
        output = dec_out[:, :, self.target_idx]  # [B, H]
        
        return output
