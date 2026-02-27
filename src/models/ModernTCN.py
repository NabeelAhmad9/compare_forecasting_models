"""
ModernTCN Forecaster.

Modern Temporal Convolutional Network with large-kernel reparameterization,
multi-stage downsampling, depthwise grouped convolutions, and optional
series decomposition. Supports RevIN normalization and structural reparameterization.

Architecture preserves exact tensor algebra, padding rules, grouped convolution
semantics, and (B, M, D, N) tensor flow patterns from the original implementation.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster
from src.models import register_model
from src.models.layers.RevIN import RevIN
from src.models.layers.ModernTCN_Layer import series_decomp, Flatten_Head


class LayerNorm(nn.Module):
    """Layer normalization module with (B, M, D, N) tensor reshaping.
    
    Preserves exact tensor flow: (B, M, D, N) -> permute -> reshape -> norm -> reshape -> permute.
    """

    def __init__(self, channels: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization preserving (B, M, D, N) flow.
        
        Args:
            x: Input tensor of shape (B, M, D, N).
            
        Returns:
            Normalized tensor of shape (B, M, D, N).
        """
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def get_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    bias: bool,
) -> nn.Conv1d:
    """Create a 1D convolution layer."""
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_bn(channels: int) -> nn.BatchNorm1d:
    """Create a 1D batch normalization layer."""
    return nn.BatchNorm1d(channels)


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: Optional[int],
    groups: int,
    dilation: int = 1,
    bias: bool = False,
) -> nn.Sequential:
    """Create a convolution-batchnorm sequential block."""
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
    )
    result.add_module("bn", get_bn(out_channels))
    return result


def fuse_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse convolution and batch normalization weights for reparameterization.
    
    Args:
        conv: Convolution layer.
        bn: Batch normalization layer.
        
    Returns:
        Tuple of (fused_kernel, fused_bias).
    """
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    """Large-kernel depthwise convolution with structural reparameterization.
    
    Combines large kernel convolution with optional small kernel branch.
    Can merge branches during inference for efficiency.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Large kernel size.
        stride: Convolution stride.
        groups: Number of groups for grouped convolution.
        small_kernel: Small kernel size for additional branch (optional).
        small_kernel_merged: Whether small kernel is already merged.
        nvars: Number of variables (for multivariate series).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: Optional[int],
        small_kernel_merged: bool = False,
        nvars: int = 7,
    ) -> None:
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=False,
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                    bias=False,
                )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional dual-branch convolution.
        
        Args:
            inputs: Input tensor (B, C, L).
            
        Returns:
            Output tensor (B, C, L).
        """
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(
        self,
        x: torch.Tensor,
        pad_length_left: int,
        pad_length_right: int,
        pad_values: float = 0,
    ) -> torch.Tensor:
        """Pad 1D kernel tensor on both edges.
        
        Args:
            x: Kernel tensor (D_out, D_in, kernel_size).
            pad_length_left: Left padding length.
            pad_length_right: Right padding length.
            pad_values: Padding value.
            
        Returns:
            Padded tensor.
        """
        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left)
            pad_right = torch.zeros(D_out, D_in, pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left, x], dim=-1)
        x = torch.cat([x, pad_right], dim=-1)
        return x

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute equivalent kernel and bias for merged reparameterization.
        
        Returns:
            Tuple of (equivalent_kernel, equivalent_bias).
        """
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(
                small_k,
                (self.kernel_size - self.small_kernel) // 2,
                (self.kernel_size - self.small_kernel) // 2,
                0,
            )
        return eq_k, eq_b

    def merge_kernel(self) -> None:
        """Merge dual-branch structure into single convolution for inference."""
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

class Block(nn.Module):
    """ModernTCN block with depthwise large-kernel convolution and dual ConvFFN layers.
    
    Implements the core building block with (B, M, D, N) tensor flow semantics.
    Preserves exact grouped convolution and tensor reshaping patterns.
    
    Args:
        large_size: Large kernel size for depthwise convolution.
        small_size: Small kernel size for reparameterization branch.
        dmodel: Model dimension.
        dff: Feedforward dimension.
        nvars: Number of variables.
        small_kernel_merged: Whether small kernel is merged.
        drop: Dropout rate.
    """

    def __init__(
        self,
        large_size: int,
        small_size: Optional[int],
        dmodel: int,
        dff: int,
        nvars: int,
        small_kernel_merged: bool = False,
        drop: float = 0.1,
    ) -> None:
        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            groups=nvars * dmodel,
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
            nvars=nvars,
        )
        self.norm = nn.BatchNorm1d(dmodel)

        # ConvFFN1
        self.ffn1pw1 = nn.Conv1d(
            in_channels=nvars * dmodel,
            out_channels=nvars * dff,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=nvars,
        )
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(
            in_channels=nvars * dff,
            out_channels=nvars * dmodel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=nvars,
        )
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # ConvFFN2 (currently unused in forward but preserved)
        self.ffn2pw1 = nn.Conv1d(
            in_channels=nvars * dmodel,
            out_channels=nvars * dff,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=dmodel,
        )
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(
            in_channels=nvars * dff,
            out_channels=nvars * dmodel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=dmodel,
        )
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with exact (B, M, D, N) tensor flow.
        
        Args:
            x: Input tensor (B, M, D, N).
            
        Returns:
            Output tensor (B, M, D, N).
        """
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = input + x
        return x


class Stage(nn.Module):
    """Multi-block stage in ModernTCN hierarchy.
    
    Args:
        ffn_ratio: Ratio of feedforward dimension to model dimension.
        num_blocks: Number of blocks in this stage.
        large_size: Large kernel size.
        small_size: Small kernel size.
        dmodel: Model dimension.
        dw_model: Depthwise model dimension (unused but kept for interface).
        nvars: Number of variables.
        small_kernel_merged: Whether small kernels are merged.
        drop: Dropout rate.
    """

    def __init__(
        self,
        ffn_ratio: int,
        num_blocks: int,
        large_size: int,
        small_size: Optional[int],
        dmodel: int,
        dw_model: int,
        nvars: int,
        small_kernel_merged: bool = False,
        drop: float = 0.1,
    ) -> None:
        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(
                large_size=large_size,
                small_size=small_size,
                dmodel=dmodel,
                dff=d_ffn,
                nvars=nvars,
                small_kernel_merged=small_kernel_merged,
                drop=drop,
            )
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all blocks.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        for blk in self.blocks:
            x = blk(x)

        return x


class ModernTCN(nn.Module):
    """Core ModernTCN architecture with multi-stage downsampling and large-kernel convolutions.
    
    Preserves exact tensor algebra, padding rules, grouped convolution semantics,
    and (B, M, D, N) tensor flow patterns from the original implementation.
    
    Args:
        patch_size: Patch size for initial embedding.
        patch_stride: Stride for patching.
        stem_ratio: Stem layer ratio (unused but kept for interface).
        downsample_ratio: Downsampling ratio between stages.
        ffn_ratio: Feedforward expansion ratio.
        num_blocks: Number of blocks per stage (list).
        large_size: Large kernel sizes per stage (list).
        small_size: Small kernel sizes per stage (list).
        dims: Model dimensions per stage (list).
        dw_dims: Depthwise dimensions per stage (list, unused but kept).
        nvars: Number of variables.
        small_kernel_merged: Whether small kernels are already merged.
        backbone_dropout: Dropout rate in backbone.
        head_dropout: Dropout rate in head.
        use_multi_scale: Whether to use multi-scale head.
        revin: Whether to use RevIN normalization.
        affine: Whether RevIN uses affine parameters.
        subtract_last: Whether RevIN subtracts last value.
        freq: Frequency encoding (unused).
        seq_len: Input sequence length.
        c_in: Number of input channels.
        individual: Whether to use individual heads per variable.
        target_window: Forecast horizon.
    """

    def __init__(
        self,
        patch_size: int,
        patch_stride: int,
        stem_ratio: int,
        downsample_ratio: int,
        ffn_ratio: int,
        num_blocks: List[int],
        large_size: List[int],
        small_size: List[Optional[int]],
        dims: List[int],
        dw_dims: List[int],
        nvars: int,
        small_kernel_merged: bool = False,
        backbone_dropout: float = 0.1,
        head_dropout: float = 0.1,
        use_multi_scale: bool = True,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        freq: Optional[str] = None,
        seq_len: int = 512,
        c_in: int = 7,
        individual: bool = False,
        target_window: int = 96,
    ) -> None:
        super(ModernTCN, self).__init__()

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Stem layer & downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0]),
        )
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(
                        dims[i],
                        dims[i + 1],
                        kernel_size=downsample_ratio,
                        stride=downsample_ratio,
                    ),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # Backbone
        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(
                ffn_ratio,
                num_blocks[stage_idx],
                large_size[stage_idx],
                small_size[stage_idx],
                dmodel=dims[stage_idx],
                dw_model=dw_dims[stage_idx],
                nvars=nvars,
                small_kernel_merged=small_kernel_merged,
                drop=backbone_dropout,
            )
            self.stages.append(layer)

        # Head
        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[self.num_stage - 1]

        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = Flatten_Head(
                self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout
            )
        else:
            if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
            else:
                self.head_nf = d_model * (
                    patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1
                )
            self.head = Flatten_Head(
                self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout
            )

    def forward_feature(
        self, x: torch.Tensor, te: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract features through multi-stage backbone.
        
        Preserves exact padding rules and tensor reshaping patterns.
        
        Args:
            x: Input tensor (B, M, L).
            te: Optional time encoding (unused).
            
        Returns:
            Feature tensor (B, M, D, N).
        """
        B, M, L = x.shape
        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def forward(
        self, x: torch.Tensor, te: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with RevIN normalization.
        
        Preserves exact permutation order and normalization/denormalization flow.
        
        Args:
            x: Input tensor (B, M, L).
            te: Optional time encoding (unused).
            
        Returns:
            Output tensor (B, M, target_window) or (B, target_window) if not individual.
        """
        # Instance norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "norm")
            x = x.permute(0, 2, 1)
        x = self.forward_feature(x, te)
        x = self.head(x)
        # De-instance norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "denorm")
            x = x.permute(0, 2, 1)
        return x

    def structural_reparam(self) -> None:
        """Merge reparameterizable branches for inference efficiency."""
        for m in self.modules():
            if hasattr(m, "merge_kernel"):
                m.merge_kernel()


@register_model("ModernTCN")
class ModernTCNForecaster(BaseForecaster):
    """ModernTCN forecasting model with framework integration.
    
    Adapts the ModernTCN architecture for univariate forecasting via target selection.
    Preserves internal architecture, decomposition pathways, and RevIN normalization 
    exactly as implemented in the original ModernTCN.
    
    Supports optional series decomposition into residual and trend components,
    each processed by separate ModernTCN instances and merged at output.

    Defaults and HPO for this class are constrained to the medium-capacity
    regime (220k–250k trainable parameters). See configs/models/ModernTCN/search_space.yaml
    for the allowed HPO choices.

    Args:
        input_size: Number of input features (default: 5 for OHLCV).
        window_size: Length of input sequence.
        horizon: Number of steps to forecast.
        target_idx: Index of target variable to forecast (default: 3 for Close).
        patch_size: Patch size for initial embedding.
        patch_stride: Stride for patching.
        stem_ratio: Stem layer ratio (unused, kept for interface).
        downsample_ratio: Downsampling ratio between stages.
        ffn_ratio: Feedforward expansion ratio.
        num_blocks: Number of blocks per stage (list).
        large_size: Large kernel sizes per stage (list).
        small_size: Small kernel sizes per stage (list).
        dims: Model dimensions per stage (list).
        dw_dims: Depthwise dimensions per stage (list, unused).
        small_kernel_merged: Whether small kernels are already merged.
        dropout: Dropout rate in backbone.
        head_dropout: Dropout rate in head.
        use_multi_scale: Whether to use multi-scale head.
        revin: Whether to use RevIN normalization.
        affine: Whether RevIN uses affine parameters.
        subtract_last: Whether RevIN subtracts last value.
        individual: Whether to use individual heads per variable.
        decomposition: Whether to use series decomposition.
        kernel_size: Kernel size for decomposition moving average.
    """

    def __init__(
        self,
        input_size: int = 5,
        window_size: int = 168,
        horizon: int = 4,
        target_idx: int = 3,
        # Architecture hyperparameters
        patch_size: int = 16,
        patch_stride: int = 8,
        stem_ratio: int = 8,
        downsample_ratio: int = 2,
        ffn_ratio: int = 1,
        num_blocks: List[int] = None,
        large_size: List[int] = None,
        small_size: List[Optional[int]] = None,
        dims: List[int] = None,
        dw_dims: List[int] = None,
        small_kernel_merged: bool = False,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        use_multi_scale: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        individual: bool = False,
        # Decomposition (disabled by default for medium regime)
        decomposition: bool = False,
        kernel_size: int = 25,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name="ModernTCN",
            input_size=input_size,
            window_size=window_size,
            horizon=horizon,
            target_idx=target_idx,
            patch_size=patch_size,
            patch_stride=patch_stride,
            stem_ratio=stem_ratio,
            downsample_ratio=downsample_ratio,
            ffn_ratio=ffn_ratio,
            num_blocks=num_blocks,
            large_size=large_size,
            small_size=small_size,
            dims=dims,
            dw_dims=dw_dims,
            small_kernel_merged=small_kernel_merged,
            dropout=dropout,
            head_dropout=head_dropout,
            use_multi_scale=use_multi_scale,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            individual=individual,
            decomposition=decomposition,
            kernel_size=kernel_size,
            **kwargs,
        )

        self.target_idx = target_idx

        # Default hyperparameters (medium-capacity regime)
        if num_blocks is None:
            # keep total blocks <= 4 (e.g. [1,1,1])
            num_blocks = [1, 1, 1]
        if large_size is None:
            # keep original large kernels (does not materially change param count)
            large_size = [51, 49, 47]
        if small_size is None:
            small_size = [5, 5, 5]
        if dims is None:
            # medium-capacity widths; last stage <= 128
            dims = [32, 64, 96]
        if dw_dims is None:
            dw_dims = [32, 64, 96]

        # Basic hyperparameter sanity checks (hard constraints)
        if not (1 <= ffn_ratio <= 2):
            raise ValueError("ffn_ratio must be 1 or 2 for the medium-capacity regime")
        if sum(num_blocks) > 4:
            raise ValueError(f"Total number of blocks ({sum(num_blocks)}) exceeds allowed max (4)")
        if len(dims) != len(num_blocks):
            raise AssertionError(
                f"dims length ({len(dims)}) must match num_blocks length ({len(num_blocks)})"
            )
        if dims[-1] > 128:
            raise ValueError(f"Final stage dimension ({dims[-1]}) must be <= 128")

        # Patch parameters must be consistent to avoid negative padding in forward_feature
        if patch_stride > patch_size:
            raise ValueError(
                f"Invalid patch configuration: patch_stride ({patch_stride}) must be <= patch_size ({patch_size}). "
                "This prevents negative padding or invalid tensor shapes."
            )

        # Validate that all stage-related lists have the same length
        num_stages = len(num_blocks)
        assert len(large_size) == num_stages, (
            f"large_size length ({len(large_size)}) must match num_blocks length ({num_stages})"
        )
        assert len(small_size) == num_stages, (
            f"small_size length ({len(small_size)}) must match num_blocks length ({num_stages})"
        )
        assert len(dw_dims) == num_stages, (
            f"dw_dims length ({len(dw_dims)}) must match num_blocks length ({num_stages})"
        )

        # Store configuration
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.stem_ratio = stem_ratio
        self.downsample_ratio = downsample_ratio
        self.ffn_ratio = ffn_ratio
        self.num_blocks = num_blocks
        self.large_size = large_size
        self.small_size = small_size
        self.dims = dims
        self.dw_dims = dw_dims
        self.nvars = input_size
        self.small_kernel_merged = small_kernel_merged
        self.drop_backbone = dropout
        self.drop_head = head_dropout
        self.use_multi_scale = use_multi_scale
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.individual = individual
        self.decomposition = decomposition
        self.kernel_size = kernel_size

        # Build ModernTCN instance(s)
        if self.decomposition:
            self.decomp_module = series_decomp(self.kernel_size)
            self.model_res = ModernTCN(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                stem_ratio=self.stem_ratio,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                dw_dims=self.dw_dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.drop_backbone,
                head_dropout=self.drop_head,
                use_multi_scale=self.use_multi_scale,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                freq=None,
                seq_len=window_size,
                c_in=input_size,
                individual=self.individual,
                target_window=horizon,
            )
            self.model_trend = ModernTCN(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                stem_ratio=self.stem_ratio,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                dw_dims=self.dw_dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.drop_backbone,
                head_dropout=self.drop_head,
                use_multi_scale=self.use_multi_scale,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                freq=None,
                seq_len=window_size,
                c_in=input_size,
                individual=self.individual,
                target_window=horizon,
            )
        else:
            self.model = ModernTCN(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                stem_ratio=self.stem_ratio,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                dw_dims=self.dw_dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.drop_backbone,
                head_dropout=self.drop_head,
                use_multi_scale=self.use_multi_scale,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                freq=None,
                seq_len=window_size,
                c_in=input_size,
                individual=self.individual,
                target_window=horizon,
            )

        # Enforce the medium-capacity trainable-parameter budget (220k - 250k)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ModernTCN parameter count: {total_params:,}")
        if total_params < 220_000 or total_params > 250_000:
            raise ValueError(
                f"ModernTCN parameter count {total_params:,} outside allowed range (220K–250K)."
            )

    def forward(
        self, x: torch.Tensor, x_mark_enc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for univariate forecasting via target selection.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, num_features).
            x_mark_enc: Optional time markers (unused, for interface compatibility).
            
        Returns:
            Predictions of shape (batch_size, horizon) for target variable only.
            
        Raises:
            AssertionError: If input shape or dtype is invalid.
        """
        # Input validation
        assert x.ndim == 3, f"Expected 3D input (B, T, F), got {x.ndim}D"
        batch_size, seq_len, num_features = x.shape
        assert seq_len == self.window_size, (
            f"Expected sequence length {self.window_size}, got {seq_len}"
        )
        assert num_features == self.input_size, (
            f"Expected {self.input_size} features (OHLCV), got {num_features}"
        )
        assert self.target_idx < num_features, (
            f"target_idx {self.target_idx} out of range for {num_features} features"
        )

        # Ensure deterministic behavior in eval mode
        if not self.training:
            torch.use_deterministic_algorithms(mode=False)  # ModernTCN may use non-deterministic ops

        # Time encoding (unused but preserved for interface)
        te = None

        # Forward pass through ModernTCN
        if self.decomposition:
            # Decomposition pathway: (B, T, F) -> residual and trend
            res_init, trend_init = self.decomp_module(x)
            # Permute to (B, F, T) for ModernTCN internal forward
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)
            # Forward through dual branches
            res = self.model_res(res_init, te)  # (B, F, horizon)
            trend = self.model_trend(trend_init, te)  # (B, F, horizon)
            # Merge branches
            output = res + trend  # (B, F, horizon)
        else:
            # Single pathway: (B, T, F) -> (B, F, T)
            x_perm = x.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)
            # Forward through single model
            output = self.model(x_perm, te)  # (B, F, horizon) or (B, horizon) depending on head

        # Target variable selection for univariate forecasting
        # Output from ModernTCN.head is (B, nvars, horizon) in most cases
        # Even with individual=False, the Flatten_Head outputs (B, nvars, horizon)
        if output.ndim == 3:
            # Output is (B, nvars, horizon) - select target variable
            assert output.shape[1] == self.input_size, (
                f"Expected {self.input_size} channels in output, got {output.shape[1]}"
            )
            target_output = output[:, self.target_idx, :]  # (B, horizon)
        elif output.ndim == 2:
            # Output is already (B, horizon)
            target_output = output
        else:
            raise RuntimeError(f"Unexpected output dimensionality: {output.ndim}D")

        # Final shape validation
        assert target_output.shape == (batch_size, self.horizon), (
            f"Expected output shape ({batch_size}, {self.horizon}), got {target_output.shape}"
        )
        assert target_output.dtype == x.dtype, (
            f"Output dtype {target_output.dtype} does not match input dtype {x.dtype}"
        )

        return target_output

    def structural_reparam(self) -> None:
        """Merge reparameterizable branches for inference efficiency."""
        if self.decomposition:
            self.model_res.structural_reparam()
            self.model_trend.structural_reparam()
        else:
            self.model.structural_reparam()