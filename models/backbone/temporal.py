"""Multi-temporal conditioning for the SEN12MS-CR-TS bridge.

The bridge interpolates between the clean target and ONE cloudy image, so the
dataset's least-cloudy-first composite stays the bridge endpoint.  These
modules additionally give the U-Net every input date, which the composite
collapses away.

``temporal_mode="attention"``
    A shared per-date encoder embeds each input date (its optical image, its
    cloud mask, its SAR, and its acquisition date relative to the target) at
    the first ``levels`` U-Net scales.  At each of those scales every pixel of
    the optical encoder attends over the T dates.  Cloudy pixels of a date are
    excluded from the attention, and a learned null token takes the weight
    where every date is cloudy.  The result is added to the encoder features
    through a zero-initialised projection, so a fresh model starts out
    identical to one without temporal conditioning.

``temporal_mode="concat"``
    The stack, its masks and the date offsets are concatenated to the encoder
    input (baseline for the attention variant).

``sar_change`` (attention mode, needs the target date's SAR)
    For each input date, the change in backscatter between that date and the
    target date (VV, VH; dB, after a 5×5 box filter against speckle).  It is
    an extra input of the per-date encoder and, through a zero-initialised
    1×1 conv per level, an attention bias: where the radar says the surface
    changed, the model can learn to trust that date's optical pixels less.

``date_conditioning`` (independent of the mode)
    The target's day-of-year is embedded and added to the diffusion timestep
    embedding, so every time-conditioned NAFBlock (encoder, bottleneck and
    decoder) knows the season being reconstructed.  The last layer is
    zero-initialised, so a fresh model starts out identical to the baseline.

The per-date features do not depend on the diffusion time, so the bridge
sampler encodes them once per sample and reuses them at every step.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafblock import NAFBlock

# Batch keys the temporal modes read (produced by SEN12MSCRTSDataset)
TEMPORAL_KEYS = ("cloudy_stack", "input_masks", "input_dt", "input_doy", "target_doy")

# Per-level encoder output: features (B, T, C, h, w) and attention bias (B, T, 1, h, w)
TemporalLevel = Tuple[torch.Tensor, torch.Tensor]


def _channel_norm(x: torch.Tensor) -> torch.Tensor:
    """LayerNorm over channels at every pixel, (B, C, H, W) → (B, C, H, W)."""
    return F.layer_norm(x.permute(0, 2, 3, 1), x.shape[1:2]).permute(0, 3, 1, 2)


def concat_features(temporal: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concat-mode input channels: stack, masks, date offsets, target day-of-year."""
    stack, masks = temporal["cloudy_stack"], temporal["input_masks"]       # (B, T, C, H, W), (B, T, 1, H, W)
    B, T, _, H, W = stack.shape
    doy = 2 * math.pi * temporal["target_doy"]
    scalars = torch.cat([temporal["input_dt"], doy.sin()[:, None], doy.cos()[:, None]], dim=1)   # (B, T + 2)
    return torch.cat([stack.flatten(1, 2), masks.flatten(1, 2),
                      scalars[:, :, None, None].expand(B, T + 2, H, W).to(stack.dtype)], dim=1)


def concat_channels(in_channels_optical: int, n_input: int) -> int:
    return n_input * (in_channels_optical + 1) + n_input + 2


class DateEmbedding(nn.Module):
    """Acquisition date of an input relative to the target → (…, dim).

    Encodes the signed offset to the target (years; Fourier features with
    periods from 2 years down to ~6 days) and the day-of-year of both the input
    and the target date (phenology).
    """

    def __init__(self, dim: int, n_freq: int = 8) -> None:
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq, dtype=torch.float32),
                             persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq + 4, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, dt: torch.Tensor, doy_in: torch.Tensor, doy_tgt: torch.Tensor) -> torch.Tensor:
        a = dt[..., None].float() * self.freqs
        d_in, d_tgt = 2 * math.pi * doy_in.float(), 2 * math.pi * doy_tgt.float()
        feats = torch.cat([a.sin(), a.cos(), torch.stack([d_in.sin(), d_in.cos(), d_tgt.sin(), d_tgt.cos()], -1)], -1)
        return self.mlp(feats)


class TargetDateEmbedding(nn.Module):
    """Target day-of-year (B,) → (B, dim), added to the timestep embedding.

    Harmonics of the annual cycle; the output layer starts at zero.
    """

    def __init__(self, dim: int, n_harmonics: int = 4) -> None:
        super().__init__()
        self.register_buffer("k", 2 * math.pi * torch.arange(1, n_harmonics + 1, dtype=torch.float32),
                             persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * n_harmonics, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        a = doy.float()[:, None] * self.k
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=-1))


class TemporalEncoder(nn.Module):
    """Shared per-date encoder: (optical_t, mask_t, SAR_t, date_t) → features at each scale.

    Args:
        in_channels_optical: Optical bands per date.
        sar_per_date:        SAR channels per date (2 = VV+VH).
        channels:            Feature width per encoder level (matches the U-Net's first levels).
        date_dim:            Width of the date embedding.
        sar_change:          Add the per-date SAR change to the target date (input
                             and attention bias); the SAR stack must end with the
                             target date's pair.
    """

    def __init__(self, in_channels_optical: int, sar_per_date: int, channels: Sequence[int],
                 date_dim: int = 128, sar_change: bool = False) -> None:
        super().__init__()
        self.sar_per_date = sar_per_date
        self.sar_change = sar_change
        n_change = sar_per_date if sar_change else 0
        self.input = nn.Conv2d(in_channels_optical + 1 + sar_per_date + n_change, channels[0], 1)
        if sar_change:
            self.change_gate = nn.ModuleList([nn.Conv2d(sar_per_date, 1, 1) for _ in channels])
            for g in self.change_gate:
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)
        self.date_emb = DateEmbedding(date_dim)
        self.date_proj = nn.ModuleList([nn.Linear(date_dim, c) for c in channels])
        self.blocks = nn.ModuleList([NAFBlock(c) for c in channels])
        self.downs = nn.ModuleList([nn.Conv2d(channels[i], channels[i + 1], 2, stride=2)
                                    for i in range(len(channels) - 1)])

    def forward(self, temporal: Dict[str, torch.Tensor], sar: torch.Tensor) -> List[TemporalLevel]:
        stack, masks = temporal["cloudy_stack"], temporal["input_masks"]
        B, T, _, H, W = stack.shape
        s = self.sar_per_date
        # Input dates' SAR come first in the SAR stack; a target-date pair (if any) follows
        sar_t = sar[:, : T * s].reshape(B, T, s, H, W).flatten(0, 1)                # (B·T, s, H, W)
        parts = [stack.flatten(0, 1), masks.flatten(0, 1).to(stack.dtype), sar_t.to(stack.dtype)]
        change = None
        if self.sar_change:
            if sar.shape[1] < (T + 1) * s:
                raise ValueError("sar_change needs the target date's SAR (data.ts.include_target_sar)")
            smooth = lambda x: F.avg_pool2d(x.float(), 5, stride=1, padding=2, count_include_pad=False)
            target_sar = smooth(sar[:, T * s:(T + 1) * s]).repeat_interleave(T, dim=0)
            # SAR is normalised as (dB + 25) / 25, so 25·Δ is the change in dB; scale to dB / 10
            change = 2.5 * (smooth(sar_t) - target_sar)                              # (B·T, s, H, W)
            parts.append(change.to(stack.dtype))
        x = torch.cat(parts, dim=1)
        e = self.date_emb(temporal["input_dt"], temporal["input_doy"],
                          temporal["target_doy"][:, None].expand(B, T)).flatten(0, 1)   # (B·T, D)
        cloudy = masks.flatten(0, 1).float()                                         # (B·T, 1, H, W)
        change_mag = change.abs() if change is not None else None

        h = self.input(x)
        levels: List[TemporalLevel] = []
        for i, blk in enumerate(self.blocks):
            if i > 0:
                h = self.downs[i - 1](h)
                cloudy = F.avg_pool2d(cloudy, 2)
                if change_mag is not None:
                    change_mag = F.avg_pool2d(change_mag, 2)
            h = blk(h + self.date_proj[i](e).to(h.dtype)[:, :, None, None])
            # log(clear fraction): −∞-like for fully cloudy pixels, 0 for clear ones
            bias = torch.log((1.0 - cloudy).clamp_min(1e-4))
            if change_mag is not None:
                bias = bias + self.change_gate[i](change_mag)    # learned trust from radar change
            levels.append((h.unflatten(0, (B, T)), bias.unflatten(0, (B, T))))
        return levels


class TemporalFusion(nn.Module):
    """Per-pixel multi-head attention from the optical encoder over the T input dates.

    The null token's value depends on the conditioning vector (timestep and,
    with date conditioning, target date), so pixels cloudy in every date still
    receive the target season.
    """

    def __init__(self, channels: int, num_heads: int, cond_dim: int) -> None:
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.heads, self.head_dim = num_heads, channels // num_heads
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Conv2d(channels, 2 * channels, 1)
        self.null_k = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.null_v = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.null_v_cond = nn.Linear(cond_dim, channels)
        nn.init.zeros_(self.null_v_cond.weight)
        nn.init.zeros_(self.null_v_cond.bias)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, feats: torch.Tensor, bias: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """h: (B, C, H, W); feats: (B, T, C, H, W); bias: (B, T, 1, H, W); cond: (B, D) → (B, C, H, W)."""
        B, T, C, H, W = feats.shape
        nh, d = self.heads, self.head_dim
        q = self.q(_channel_norm(h)).view(B, 1, nh, d, H, W)
        k, v = self.kv(_channel_norm(feats.flatten(0, 1))).view(B, T, 2, nh, d, H, W).unbind(2)
        scale = d ** -0.5
        logits = (q * k).sum(3) * scale + bias.to(q.dtype)                        # (B, T, nh, H, W)
        null = (q[:, 0] * self.null_k.to(q.dtype)[None, :, :, None, None]).sum(2) * scale   # (B, nh, H, W)
        w = torch.cat([logits, null[:, None]], dim=1).float().softmax(dim=1).to(v.dtype)    # (B, T+1, nh, H, W)
        null_v = self.null_v[None] + self.null_v_cond(cond.float()).view(B, nh, d)      # (B, nh, d)
        out = (w[:, :T, :, None] * v).sum(1) + w[:, T, :, None] * null_v.to(v.dtype)[:, :, :, None, None]
        return h + self.proj(out.reshape(B, C, H, W))
