"""NAFNet building blocks (Simple Baselines for Image Restoration, Chen et al. 2022).

NAFBlock replaces:
  - Batch/Layer Norm  →  Layer Norm (pre-norm)
  - ReLU/GELU         →  SimpleGate (element-wise channel split)
  - SE attention      →  Simplified Channel Attention (SCA, no sigmoid)

Time-embedding injection is added for diffusion timestep conditioning:
  after the first LayerNorm, a projected time vector is broadcast-added to
  the spatial features before the depthwise conv branch.

References:
  Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
  https://arxiv.org/abs/2204.04676
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class SimpleGate(nn.Module):
    """Split along the channel axis; return the elementwise product.

    Input : (B, 2C, H, W)
    Output: (B,  C, H, W)   — X1 * X2
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Global average pool → 1×1 conv → scale.

    No sigmoid or activation — the 1×1 conv learns per-channel scalars
    directly, keeping the block activation-free.

    Input / Output: (B, C, H, W)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x)) * x


# ---------------------------------------------------------------------------
# NAFBlock
# ---------------------------------------------------------------------------

class NAFBlock(nn.Module):
    """Single NAFNet block with optional diffusion timestep conditioning.

    Spatial (MBConv) branch  — applied first:
        LN → conv1×1(C→2C) → DWConv3×3(2C→2C) → SimpleGate(2C→C)
           → SCA(C) → conv1×1(C→C) → β-scaled residual

    Channel (FFN) branch  — applied second:
        LN → conv1×1(C→2C) → SimpleGate(2C→C) → conv1×1(C→C)
           → γ-scaled residual

    Time-embedding injection (when ``time_emb_dim`` is given):
        A two-layer MLP (SiLU → Linear) projects ``time_emb`` (B, D) →
        (B, C), which is broadcast-added to the features immediately
        after the first LayerNorm and before the depthwise conv.

    Args:
        channels:     Number of feature channels C.
        dw_expand:    Channel expansion factor for the spatial branch (default 2).
        ffn_expand:   Channel expansion factor for the FFN branch (default 2).
        drop_out:     Dropout probability applied after each branch (default 0).
        time_emb_dim: Dimensionality of the incoming time embedding.
                      Pass ``None`` (default) to disable conditioning.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out: float = 0.0,
        time_emb_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        dw_ch  = channels * dw_expand   # 2C
        ffn_ch = channels * ffn_expand  # 2C

        # --- Spatial (MBConv) branch ---
        self.norm1 = nn.LayerNorm(channels)
        # Pointwise expand: C → 2C
        self.conv1 = nn.Conv2d(channels, dw_ch, 1, bias=True)
        # Depthwise 3×3 on the full expanded width (2C → 2C)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch, bias=True)
        # SimpleGate halves channel count: 2C → C
        self.sg1   = SimpleGate()
        # Channel attention on C features
        self.sca   = SimplifiedChannelAttention(dw_ch // 2)
        # Pointwise compress: C → C
        self.conv3 = nn.Conv2d(dw_ch // 2, channels, 1, bias=True)

        # --- Timestep conditioning ---
        # Activated before the spatial branch; injects global time information.
        if time_emb_dim is not None:
            self.time_mlp: Optional[nn.Module] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, channels),
            )
        else:
            self.time_mlp = None

        # --- Channel (FFN) branch ---
        self.norm2 = nn.LayerNorm(channels)
        # Pointwise expand: C → 2C
        self.conv4 = nn.Conv2d(channels, ffn_ch, 1, bias=True)
        # SimpleGate halves: 2C → C
        self.sg2   = SimpleGate()
        # Pointwise compress: C → C
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, 1, bias=True)

        self.dropout = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()

        # Learnable per-channel residual scaling (initialised to 1, per NAFNet)
        self.beta  = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (B, C, H, W) feature map.
            time_emb: (B, time_emb_dim) diffusion timestep embedding.
                      Ignored when the block was built without ``time_emb_dim``.
        Returns:
            (B, C, H, W) — same spatial resolution, same channel count.
        """
        # ---- Spatial branch -----------------------------------------------
        # LayerNorm operates in channel-last; permute in/out.
        h = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Inject timestep conditioning (broadcast over H, W).
        if self.time_mlp is not None and time_emb is not None:
            t = self.time_mlp(time_emb)          # (B, C)
            h = h + t[:, :, None, None]          # (B, C, H, W)

        h = self.conv1(h)   # C  → 2C
        h = self.conv2(h)   # 2C → 2C  (depthwise)
        h = self.sg1(h)     # 2C → C   (SimpleGate)
        h = self.sca(h)     # C  → C   (channel attention)
        h = self.conv3(h)   # C  → C
        h = self.dropout(h)
        x = x + h * self.beta

        # ---- FFN branch ---------------------------------------------------
        h = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.conv4(h)   # C  → 2C
        h = self.sg2(h)     # 2C → C   (SimpleGate)
        h = self.conv5(h)   # C  → C
        h = self.dropout(h)
        x = x + h * self.gamma

        return x


# ---------------------------------------------------------------------------
# NAFNet — stacked encoder-decoder for standalone image restoration
# (time conditioning is not used here; NAFBlock defaults time_emb=None)
# ---------------------------------------------------------------------------

class NAFNet(nn.Module):
    """Full NAFNet U-shape image restoration network.

    Args:
        in_channels:     Input image channels.
        out_channels:    Output channels (equals ``in_channels`` by default).
        width:           Base feature width.
        enc_blks:        NAFBlocks per encoder level.
        dec_blks:        NAFBlocks per decoder level.
        middle_blk_num:  NAFBlocks in the bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 13,
        out_channels: Optional[int] = None,
        width: int = 32,
        enc_blks: Optional[List[int]] = None,
        dec_blks: Optional[List[int]] = None,
        middle_blk_num: int = 12,
    ) -> None:
        super().__init__()
        out_channels   = out_channels or in_channels
        enc_blks       = enc_blks or [2, 2, 4, 8]
        dec_blks       = dec_blks or [2, 2, 2, 2]

        self.intro  = nn.Conv2d(in_channels,  width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        self.ups      = nn.ModuleList()

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
                    nn.PixelShuffle(2),          # chan → chan//2, ×2 spatial
                )
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = self.intro(x)

        enc_skips: List[torch.Tensor] = []
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

        return self.ending(h) + x[:, : self.ending.out_channels]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 64, 32, 32
    T = 128

    block = NAFBlock(channels=C, time_emb_dim=T)
    x     = torch.randn(B, C, H, W)
    t_emb = torch.randn(B, T)

    out = block(x, t_emb)
    assert out.shape == (B, C, H, W), f"Unexpected shape: {out.shape}"
    print(f"NAFBlock smoke-test passed  |  input {tuple(x.shape)}  →  output {tuple(out.shape)}")

    # Without time embedding (backward-compat)
    block_no_t = NAFBlock(channels=C)
    out2 = block_no_t(x)
    assert out2.shape == (B, C, H, W)
    print(f"NAFBlock (no time emb) passed  |  output {tuple(out2.shape)}")
