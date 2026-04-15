"""Full U-Net assembling NAFBlocks, Bidirectional Mamba SSM, and SAR Fusion Blocks.

The architecture:
  Encoder path  : NAFBlock stacks with stride-2 down-sampling at each level.
  Bottleneck    : BidirectionalMamba (global sequence modelling on flattened tokens).
  Decoder path  : Up-sample + skip-concat + 1×1 projection + NAFBlock stacks
                  + SARFusionBlock (cross-attention with SAR features).
  Conditioning  : Timestep t injected into every NAFBlock via its internal MLP.

Bug-fixes vs. original
-----------------------
1. Decoder channel mismatch: after torch.cat([up(h), skip]), h has
   (ch + skip_ch) channels.  A 1×1 skip_proj reduces it back to ch
   before the NAFBlocks and SARFusionBlock — the original ``pass`` stub
   would crash on the second decoder NAFBlock.
2. Time embedding was computed but never forwarded to NAFBlocks.
   All NAFBlocks now receive t_emb; their internal SiLU→Linear MLP
   handles the projection.
3. Bottleneck nn.Sequential couldn't forward kwargs.  Changed to
   nn.ModuleList so t_emb can be passed explicitly.
4. Removed the now-redundant ``time_projs`` ModuleList (each NAFBlock
   owns its own time projection).
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
        # stride-2 conv doubles channels while halving spatial size
        self.conv = nn.Conv2d(channels, channels * 2, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # PixelShuffle(2): (B, 4C, H, W) → (B, C, 2H, 2W)
        # conv maps C → 2C; shuffle maps 2C → C/2... wait:
        # shuffle requires 4×out channels → conv must produce 4×out.
        # We want out = channels//2, so conv produces channels//2 * 4 = 2*channels.
        self.conv = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)   # (B, 2C, H, W) → (B, C//2, 2H, 2W)

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
        attn_resolutions    = attn_resolutions    or [16, 8]

        time_dim = base_channels * 4
        self.time_emb = TimeEmbedding(time_dim)

        # --- Initial projection ---
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Encoder ---
        self.encoder_blocks  = nn.ModuleList()
        self.downs           = nn.ModuleList()
        self.sar_fusions_enc = nn.ModuleList()

        enc_channels: List[int] = []
        ch = base_channels
        current_res = 256
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            # Pass time_dim so each NAFBlock can condition on t_emb
            blocks = nn.ModuleList(
                [NAFBlock(ch, drop_out=dropout, time_emb_dim=time_dim)
                 for _ in range(num_res_blocks)]
            )
            self.encoder_blocks.append(blocks)

            if current_res in attn_resolutions:
                self.sar_fusions_enc.append(SARFusionBlock(ch, cond_channels))
            else:
                self.sar_fusions_enc.append(nn.Identity())

            enc_channels.append(ch)
            self.downs.append(DownSample(ch))
            ch = out_ch
            current_res //= 2

        # --- Bottleneck: NAFBlocks + Bidirectional Mamba ---
        # ModuleList instead of Sequential so we can pass t_emb explicitly.
        self.bottleneck_in  = nn.ModuleList(
            [NAFBlock(ch, time_emb_dim=time_dim) for _ in range(2)]
        )
        self.bottleneck_mamba = BidirectionalMamba(ch, d_state=mamba_d_state)
        self.bottleneck_out = nn.ModuleList(
            [NAFBlock(ch, time_emb_dim=time_dim) for _ in range(2)]
        )

        # --- Decoder ---
        self.decoder_blocks  = nn.ModuleList()
        self.ups             = nn.ModuleList()
        self.skip_projs      = nn.ModuleList()   # 1×1 conv: (ch + skip_ch) → ch
        self.sar_fusions_dec = nn.ModuleList()

        for mult, skip_ch in zip(reversed(channel_multipliers), reversed(enc_channels)):
            self.ups.append(UpSample(ch))
            ch = ch // 2          # UpSample halves channels
            in_ch_dec = ch + skip_ch

            # Project concatenated skip+up tensor back to ch channels.
            self.skip_projs.append(nn.Conv2d(in_ch_dec, ch, 1, bias=False))

            blocks = nn.ModuleList(
                [NAFBlock(ch, drop_out=dropout, time_emb_dim=time_dim)
                 for _ in range(num_res_blocks)]
            )
            self.decoder_blocks.append(blocks)

            if current_res in attn_resolutions:
                self.sar_fusions_dec.append(SARFusionBlock(ch, cond_channels))
            else:
                self.sar_fusions_dec.append(nn.Identity())

            current_res *= 2

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
        skips: List[torch.Tensor] = []
        for blks, down, sar_fuse in zip(
            self.encoder_blocks, self.downs, self.sar_fusions_enc
        ):
            for blk in blks:
                h = blk(h, t_emb)
            if sar_feat is not None and isinstance(sar_fuse, SARFusionBlock):
                h = sar_fuse(h, sar_feat)
            skips.append(h)
            h = down(h)

        # --- Bottleneck ---
        for blk in self.bottleneck_in:
            h = blk(h, t_emb)
        h = self._mamba_forward(h)
        for blk in self.bottleneck_out:
            h = blk(h, t_emb)

        # --- Decoder ---
        for i, (up, skip_proj, blks, sar_fuse) in enumerate(
            zip(self.ups, self.skip_projs, self.decoder_blocks, self.sar_fusions_dec)
        ):
            h = up(h)
            skip = skips[-(i + 1)]
            # Align SAR spatial size if padding caused a 1-pixel mismatch
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear",
                                  align_corners=False)
            h = torch.cat([h, skip], dim=1)     # (B, ch + skip_ch, H, W)
            h = skip_proj(h)                    # (B, ch, H, W)  — channel reduction
            for blk in blks:
                h = blk(h, t_emb)
            if sar_feat is not None and isinstance(sar_fuse, SARFusionBlock):
                h = sar_fuse(h, sar_feat)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
