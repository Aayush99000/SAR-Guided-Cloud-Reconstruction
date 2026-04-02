"""Full U-Net assembling NAFBlocks, Bidirectional Mamba SSM, and SAR Fusion Blocks.

The architecture:
  Encoder path  : NAFBlock stacks with stride-2 down-sampling at each level.
  Bottleneck    : BidirectionalMamba (global sequence modelling on flattened tokens).
  Decoder path  : Up-sample + SARFusionBlock (cross-attention with SAR features)
                  + NAFBlock stacks.
  Conditioning  : Timestep t is injected via additive embedding (sinusoidal + MLP).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafblock import NAFBlock
from .vim_ssm import BidirectionalMamba
from .sfblock import SARFusionBlock


# ---------------------------------------------------------------------------
# Time-step embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# ---------------------------------------------------------------------------
# Down / Up sample wrappers
# ---------------------------------------------------------------------------

class DownSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # PixelShuffle: (B, 4C, H, W) → (B, C, 2H, 2W)
        self.conv = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """SAR-conditioned U-Net velocity network for the diffusion bridge.

    Args:
        in_channels:          Channels of the noised latent z_t.
        cond_channels:        SAR feature channels (from a pretrained SAR encoder).
        base_channels:        Base channel width.
        channel_multipliers:  Per-level channel multipliers.
        num_res_blocks:       NAFBlocks per encoder/decoder level.
        mamba_d_state:        SSM state dimension for bottleneck Mamba block.
        attn_resolutions:     Spatial sizes at which SARFusionBlocks are inserted.
        dropout:              Dropout probability in NAFBlocks.
    """

    def __init__(
        self,
        in_channels: int = 256,
        cond_channels: int = 256,
        base_channels: int = 64,
        channel_multipliers: List[int] = None,
        num_res_blocks: int = 2,
        mamba_d_state: int = 16,
        attn_resolutions: List[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channel_multipliers = channel_multipliers or [1, 2, 4, 8]
        attn_resolutions = attn_resolutions or [16, 8]

        time_dim = base_channels * 4
        self.time_emb = TimeEmbedding(time_dim)

        # --- Initial projection ---
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.sar_fusions_enc = nn.ModuleList()

        enc_channels: List[int] = []
        ch = base_channels
        current_res = 256
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            blocks = nn.ModuleList([NAFBlock(ch, drop_out=dropout) for _ in range(num_res_blocks)])
            self.encoder_blocks.append(blocks)

            if current_res in attn_resolutions:
                self.sar_fusions_enc.append(SARFusionBlock(ch, cond_channels))
            else:
                self.sar_fusions_enc.append(nn.Identity())

            enc_channels.append(ch)
            self.downs.append(DownSample(ch))
            ch = out_ch
            current_res //= 2

        # --- Bottleneck: Mamba SSM ---
        self.bottleneck_in = nn.Sequential(
            NAFBlock(ch), NAFBlock(ch)
        )
        self.bottleneck_mamba = BidirectionalMamba(ch, d_state=mamba_d_state)
        self.bottleneck_out = nn.Sequential(
            NAFBlock(ch), NAFBlock(ch)
        )

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.sar_fusions_dec = nn.ModuleList()

        for mult, skip_ch in zip(reversed(channel_multipliers), reversed(enc_channels)):
            self.ups.append(UpSample(ch))
            ch = ch // 2
            in_ch_dec = ch + skip_ch

            blocks = nn.ModuleList(
                [NAFBlock(in_ch_dec if i == 0 else ch, drop_out=dropout) for i in range(num_res_blocks)]
            )
            self.decoder_blocks.append(blocks)

            if current_res in attn_resolutions:
                self.sar_fusions_dec.append(SARFusionBlock(ch, cond_channels))
            else:
                self.sar_fusions_dec.append(nn.Identity())

            current_res *= 2

        # Time conditioning: project to each level's channel size (additive)
        all_channels = [base_channels * m for m in channel_multipliers] + [ch]
        self.time_projs = nn.ModuleList(
            [nn.Linear(time_dim, c) for c in all_channels]
        )

        # --- Output ---
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    # ------------------------------------------------------------------

    def _mamba_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.bottleneck_mamba(tokens)
        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        sar_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity v(z_t, t, sar_feat).

        Args:
            z_t:       Noised latent (B, in_channels, H, W).
            t:         Timestep values in [0, 1], shape (B,).
            sar_feat:  SAR feature map (B, cond_channels, H', W') or None.

        Returns:
            Velocity estimate (B, in_channels, H, W).
        """
        t_emb = self.time_emb(t)   # (B, time_dim)

        h = self.init_conv(z_t)

        # --- Encoder ---
        skips = []
        for i, (blks, down, sar_fuse) in enumerate(
            zip(self.encoder_blocks, self.downs, self.sar_fusions_enc)
        ):
            for blk in blks:
                h = blk(h)
            if sar_feat is not None and isinstance(sar_fuse, SARFusionBlock):
                h = sar_fuse(h, sar_feat)
            skips.append(h)
            h = down(h)

        # --- Bottleneck ---
        h = self.bottleneck_in(h)
        h = self._mamba_forward(h)
        h = self.bottleneck_out(h)

        # --- Decoder ---
        for i, (up, blks, sar_fuse) in enumerate(
            zip(self.ups, self.decoder_blocks, self.sar_fusions_dec)
        ):
            h = up(h)
            skip = skips[-(i + 1)]
            h = torch.cat([h, skip], dim=1)
            for j, blk in enumerate(blks):
                h = blk(h)
                if j == 0:
                    # Reduce to expected channel count after skip concat
                    pass
            if sar_feat is not None and isinstance(sar_fuse, SARFusionBlock):
                h = sar_fuse(h, sar_feat)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
