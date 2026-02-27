import time
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.functional import interpolate


def decor_time(func):
    def func2(*args, **kw):
        now = time.time()
        y = func(*args, **kw)
        t = time.time() - now
        print('call <{}>, time={}'.format(func.__name__, t))
        return y
    return func2


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, configs=None):
        super(AutoCorrelation, self).__init__()
        print('Autocorrelation used !')
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.agg = None
        # self.use_wavelet = configs.wavelet

    # @decor_time
    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design).
        Device- and dtype-safe implementation that avoids implicit `.cuda()`
        and Python/CUDA tensor transfers inside tight loops.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k (ensure at least 1)
        top_k = max(1, int(self.factor * math.log(length)))

        # mean correlation across batches/heads -> shape: (length,)
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # (batch, length)

        # get top-k delay indices (tensor lives on same device)
        topk_indices = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]

        # gather weights for those indices: (batch, top_k)
        weights = mean_value.index_select(dim=1, index=topk_indices)
        tmp_corr = torch.softmax(weights, dim=-1)  # (batch, top_k)

        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values)

        # use Python ints for roll offsets (safe and device-agnostic)
        offsets = topk_indices.tolist()
        for i, off in enumerate(offsets):
            pattern = torch.roll(tmp_values, -off, -1)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].view(-1, 1, 1, 1)

        return delays_agg  # size=[B, H, d, S]

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation for inference.
        Make index tensors on the same device as `values` to avoid CUDA/CPU mismatch.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        device = values.device

        # index init on same device
        init_index = torch.arange(length, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)

        # find top k
        top_k = max(1, int(self.factor * math.log(length)))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]

        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values)

        # ensure delay indices are on the same device for indexing
        for i in range(delay.shape[1]):
            idx_tensor = delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
            tmp_delay = init_index + idx_tensor.repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].view(-1, 1, 1, 1)

        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation (device-agnostic).
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        device = values.device

        # index init on same device
        init_index = torch.arange(length, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)

        # find top k
        top_k = max(1, int(self.factor * math.log(length)))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]

        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)

        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values)
        for i in range(tmp_corr.shape[-1]):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1).to(device)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[..., i].unsqueeze(-1)
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn