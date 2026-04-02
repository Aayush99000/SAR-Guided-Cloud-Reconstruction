"""NAFNet building blocks (Simple Baselines for Image Restoration, Chen et al. 2022).

NAFBlock replaces:
  - Batch/Layer Norm  →  Layer Norm (pre-norm)
  - ReLU/GELU         →  SimpleGate (element-wise channel split)
  - SE attention      →  Simplified Channel Attention (SCA)

References:
  Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
  https://arxiv.org/abs/2204.04676
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

class SimpleGate(nn.Module):
    """Split channels in half; first half gates the second half."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Global average pool → 1×1 conv → scale."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.avg_pool(x)) * x


# ---------------------------------------------------------------------------
# NAFBlock
# ---------------------------------------------------------------------------

class NAFBlock(nn.Module):
    """Single NAFNet block.

    Architecture (pre-norm):
        LayerNorm → DW-Conv → SG → SCA → skip
        LayerNorm → 1×1 Conv (up) → SG → 1×1 Conv (down) → skip

    Args:
        channels:    Number of feature channels.
        dw_expand:   Expansion ratio for the depthwise conv branch.
        ffn_expand:  Expansion ratio for the feed-forward branch.
        drop_out:    Dropout probability.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out: float = 0.0,
    ) -> None:
        super().__init__()
        dw_ch = channels * dw_expand
        ffn_ch = channels * ffn_expand

        # --- Spatial mixing ---
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_ch // 2, dw_ch // 2, 3, padding=1, groups=dw_ch // 2)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_ch // 2)
        self.conv3 = nn.Conv2d(dw_ch // 2, channels, 1, bias=True)

        # --- Channel mixing ---
        self.norm2 = nn.LayerNorm(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, 1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, 1, bias=True)

        self.dropout = nn.Dropout(drop_out) if drop_out > 0 else nn.Identity()

        # Learnable residual scaling
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial branch
        B, C, H, W = x.shape
        h = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.sg1(h)        # (B, C*dw_expand//2, H, W)
        h = self.sca(h)
        h = self.conv3(h)
        h = self.dropout(h)
        x = x + h * self.beta

        # Channel (FFN) branch
        h = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.conv4(h)
        h = self.sg2(h)
        h = self.conv5(h)
        h = self.dropout(h)
        x = x + h * self.gamma

        return x


# ---------------------------------------------------------------------------
# NAFNet (stacked encoder–decoder)
# ---------------------------------------------------------------------------

class NAFNet(nn.Module):
    """Full NAFNet image restoration network.

    Args:
        in_channels:     Input image channels.
        out_channels:    Output image channels (same as in_channels by default).
        width:           Base feature width.
        enc_blks:        Number of NAFBlocks per encoder level.
        dec_blks:        Number of NAFBlocks per decoder level.
        middle_blk_num:  Number of NAFBlocks in the bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 13,
        out_channels: int | None = None,
        width: int = 32,
        enc_blks: List[int] = None,
        dec_blks: List[int] = None,
        middle_blk_num: int = 12,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        enc_blks = enc_blks or [2, 2, 4, 8]
        dec_blks = dec_blks or [2, 2, 2, 2]

        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),   # chan → chan//2, H/W × 2
                )
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = self.intro(x)

        enc_skips = []
        h = inp
        for enc, down in zip(self.encoders, self.downs):
            h = enc(h)
            enc_skips.append(h)
            h = down(h)

        h = self.middle_blks(h)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(enc_skips)):
            h = up(h)
            h = h + skip
            h = dec(h)

        return self.ending(h) + x[:, :h.shape[1]]   # global residual
