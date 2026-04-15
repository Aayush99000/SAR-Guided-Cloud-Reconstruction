"""SAR-guided optical cloud reconstruction U-Net (DB-CR two-branch design).

Two branches run in parallel then fuse via SFBlock:

  SAR encoder       4-level downsampling backbone, outputs multi-scale
                    features F_sar_0 … F_sar_3 at decreasing spatial scales.
                    No diffusion timestep conditioning.

  Optical U-Net     Encoder–Middle–Decoder with SAR fusion at every encoder
                    level, global sequence modelling via VimBlocks at the
                    bottleneck, and timestep conditioning in every NAFBlock.

Input to the optical encoder:
    cat(x_t, x_cloudy_mean, sar)  →  (B, 2·C_opt + C_sar, H, W)

Output:
    Predicted clean image x_0  →  (B, C_opt, H, W)

Architecture at default settings  (base_channels=64, channel_mult=[1,2,4,8],
                                    256×256 input):
┌──────────────────────────────────────────────────────────────┐
│  Scale 0  H×W     ch=64   enc_0 (1 NAFBlk) + SFBlock  ─ skip_0
│  Scale 1  H/2×W/2 ch=128  enc_1 (1 NAFBlk) + SFBlock  ─ skip_1
│  Scale 2  H/4×W/4 ch=256  enc_2 (1 NAFBlk) + SFBlock  ─ skip_2
│  Scale 3  H/8×W/8 ch=512  enc_3 (28 NAFBlk) + SFBlock ─ skip_3
│  Middle                   VimBlock × 2 + NAFBlock
│  Scale 3  H/8×W/8 ch=512  cat(mid,skip_3) → proj → 1 NAFBlk
│  Scale 2  H/4×W/4 ch=256  ↑Up + cat(skip_2) → proj → 1 NAFBlk
│  Scale 1  H/2×W/2 ch=128  ↑Up + cat(skip_1) → proj → 1 NAFBlk
│  Scale 0  H×W     ch=64   ↑Up + cat(skip_0) → proj → 1 NAFBlk
│  Output  1×1 conv ch=C_opt
└──────────────────────────────────────────────────────────────┘

Reference: Ebel et al., "DBER: Cross-Modal Cloud Removal", 2022 (DB-CR).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafblock import NAFBlock
from .vim_ssm import VimBlock
from .sfblock import SARFusionBlock


# ---------------------------------------------------------------------------
# Time-step embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for scalar timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) in [0, 1]  →  (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]               # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    """Sinusoidal → 2-layer MLP → time_emb_dim."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin_emb(t))


# ---------------------------------------------------------------------------
# Down / Up sample primitives
# ---------------------------------------------------------------------------

class DownSample(nn.Module):
    """Stride-2 conv: (B, C, H, W) → (B, 2C, H/2, W/2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpSample(nn.Module):
    """PixelShuffle ×2: (B, C, H, W) → (B, C/2, 2H, 2W).

    1×1 conv maps C → 2C; PixelShuffle(2) maps 2C → C/2 at 2× resolution.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv    = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


# ---------------------------------------------------------------------------
# Bottleneck alternatives  (for ablation studies)
# ---------------------------------------------------------------------------

class BottleneckAttention(nn.Module):
    """Vanilla multi-head self-attention bottleneck (ablation baseline).

    Replaces VimSSM for the backbone ablation study:
        (b) NAFBlock + Self-Attention  →  ``bottleneck_type="attention"``

    Architecture: Pre-LN MHSA + Pre-LN FFN (GELU), both with residual connections.
    The spatial tokens are processed as a flattened sequence (B, H·W, C) then
    reshaped back to (B, C, H, W).  Complexity is O(H²W²) vs O(HW) for Mamba,
    which is why the SSM variant scales better to 256×256 patches.

    Args:
        channels:  Feature channels (must be divisible by num_heads).
        num_heads: Number of attention heads.  Default 8.
        dropout:   Dropout applied to attention weights and FFN activations.
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn   = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, time_emb=None) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)           # (B, H·W, C)
        # Pre-norm self-attention with residual
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out
        # Pre-norm FFN with residual
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SAROpticalUNet(nn.Module):
    """Two-branch SAR-guided diffusion U-Net for cloud removal.

    Args:
        in_channels_optical:  Optical image channels (4 = RGB+NIR, 13 = full S2).
        in_channels_sar:      SAR channels (2 = VV+VH).
        base_channels:        Feature width at encoder level 0.
        channel_mult:         Per-level channel multipliers (4 values).
        num_nafblocks:        NAFBlocks per encoder level (same for both branches).
                              Follows DB-CR: most capacity at the deepest level.
        num_dec_nafblocks:    NAFBlocks per decoder level.  Default: [1, 1, 1, 1].
        num_vim_blocks:       VimBlock count in the bottleneck.
        num_heads_sfblock:    SARFusionBlock heads per encoder level.
        time_emb_dim:         Timestep embedding dimension.
        dropout:              NAFBlock dropout probability (default 0 = off).
        vim_d_state:          Mamba SSM state size inside VimBlocks.
    """

    def __init__(
        self,
        in_channels_optical: int       = 4,
        in_channels_sar:     int       = 2,
        base_channels:       int       = 64,
        channel_mult:        List[int] = None,
        num_nafblocks:       List[int] = None,
        num_dec_nafblocks:   List[int] = None,
        num_vim_blocks:      int       = 2,
        num_heads_sfblock:   List[int] = None,
        time_emb_dim:        int       = 256,
        dropout:             float     = 0.0,
        vim_d_state:         int       = 16,
        bottleneck_type:     str       = "vim",
        fusion_mode:         str       = "sfblock",
    ) -> None:
        super().__init__()

        # --- Defaults (DB-CR recipe) ---
        channel_mult      = channel_mult      or [1, 2, 4, 8]
        num_nafblocks     = num_nafblocks     or [1, 1, 1, 28]
        num_dec_nafblocks = num_dec_nafblocks or [1, 1, 1, 1]
        num_heads_sfblock = num_heads_sfblock or [1, 1, 2, 4]

        if bottleneck_type not in ("vim", "attention", "none"):
            raise ValueError(
                f"bottleneck_type must be 'vim', 'attention', or 'none'; got {bottleneck_type!r}"
            )
        if fusion_mode not in ("sfblock", "early_concat", "none"):
            raise ValueError(
                f"fusion_mode must be 'sfblock', 'early_concat', or 'none'; got {fusion_mode!r}"
            )
        self.bottleneck_type = bottleneck_type
        self.fusion_mode     = fusion_mode

        N = len(channel_mult)
        assert len(num_nafblocks)     == N, "num_nafblocks length must match channel_mult"
        assert len(num_dec_nafblocks) == N, "num_dec_nafblocks length must match channel_mult"
        assert len(num_heads_sfblock) == N, "num_heads_sfblock length must match channel_mult"

        # channels[i] = feature width at encoder level i
        channels: List[int] = [base_channels * m for m in channel_mult]
        self.channels   = channels
        self.num_levels = N

        # ----------------------------------------------------------------
        # Shared time embedding  (optical branch only)
        # ----------------------------------------------------------------
        self.time_emb = TimeEmbedding(time_emb_dim)

        # ----------------------------------------------------------------
        # SAR encoder  —  no diffusion timestep conditioning
        #
        # Level i produces F_sar_i  (B, channels[i], H/2^i, W/2^i).
        # DownSample between levels 0→1, 1→2, 2→3  (3 downsamples).
        # ----------------------------------------------------------------
        self.sar_input = nn.Conv2d(in_channels_sar, channels[0], 1, bias=False)

        self.sar_enc_blocks = nn.ModuleList()
        self.sar_downs      = nn.ModuleList()

        for i in range(N):
            self.sar_enc_blocks.append(
                nn.ModuleList([NAFBlock(channels[i]) for _ in range(num_nafblocks[i])])
            )
            if i < N - 1:
                self.sar_downs.append(DownSample(channels[i]))

        # ----------------------------------------------------------------
        # Optical encoder  —  NAFBlocks + SFBlock at each level
        #
        # Input: cat(x_t, x_cloudy_mean [, sar_raw])
        #   fusion_mode="sfblock"/"early_concat": SAR concatenated to input
        #   fusion_mode="none":                   optical-only input
        # ----------------------------------------------------------------
        opt_in_ch = (
            2 * in_channels_optical + in_channels_sar
            if fusion_mode != "none"
            else 2 * in_channels_optical
        )
        self.opt_input = nn.Conv2d(opt_in_ch, channels[0], 1, bias=False)

        self.enc_blocks   = nn.ModuleList()
        self.enc_sfblocks = nn.ModuleList()
        self.enc_downs    = nn.ModuleList()

        for i in range(N):
            self.enc_blocks.append(
                nn.ModuleList([
                    NAFBlock(channels[i], drop_out=dropout, time_emb_dim=time_emb_dim)
                    for _ in range(num_nafblocks[i])
                ])
            )
            # SFBlock only used in sfblock fusion mode; identity otherwise
            if fusion_mode == "sfblock":
                self.enc_sfblocks.append(
                    SARFusionBlock(
                        d_optical = channels[i],
                        d_sar     = channels[i],
                        num_heads = num_heads_sfblock[i],
                    )
                )
            else:
                self.enc_sfblocks.append(nn.Identity())
            if i < N - 1:
                self.enc_downs.append(DownSample(channels[i]))

        # ----------------------------------------------------------------
        # Bottleneck: global context block + local NAFBlock
        # Operates at the deepest scale  (H/2^(N-1) × W/2^(N-1)).
        #
        # bottleneck_type="vim"       — VimBlock SSM (default, DB-CR design)
        # bottleneck_type="attention" — Multi-head self-attention (ablation b)
        # bottleneck_type="none"      — NAFBlock only, no global-context block (ablation c)
        # ----------------------------------------------------------------
        mid_ch = channels[-1]
        n_blks  = max(num_vim_blocks, 1) if bottleneck_type != "none" else 0

        if bottleneck_type == "vim":
            self.bottleneck_blocks = nn.ModuleList([
                VimBlock(dim=mid_ch, d_state=vim_d_state)
                for _ in range(n_blks)
            ])
        elif bottleneck_type == "attention":
            # num_heads must divide mid_ch; clamp to a safe value
            n_heads = min(8, mid_ch // 64) or 1
            self.bottleneck_blocks = nn.ModuleList([
                BottleneckAttention(mid_ch, num_heads=n_heads, dropout=dropout)
                for _ in range(n_blks)
            ])
        else:   # "none"
            self.bottleneck_blocks = nn.ModuleList()

        self.middle_naf = NAFBlock(mid_ch, time_emb_dim=time_emb_dim)

        # ----------------------------------------------------------------
        # Decoder  —  4 levels, processed from deepest (i=N-1) to shallowest (i=0)
        #
        # j=0  (i=N-1):  cat(middle_out, skip_{N-1})  —  no upsample
        # j=1  (i=N-2):  UpSample  then  cat(up, skip_{N-2})
        # j=2  (i=N-3):  UpSample  then  cat(up, skip_{N-3})
        # j=3  (i=0  ):  UpSample  then  cat(up, skip_0)
        #
        # After each cat: skip_proj reduces  2·channels[i] → channels[i].
        # dec_ups[k] = UpSample used before decoder step j=k+1 (k = 0 … N-2).
        # ----------------------------------------------------------------
        self.dec_blocks     = nn.ModuleList()
        self.dec_skip_projs = nn.ModuleList()
        self.dec_ups        = nn.ModuleList()

        for j, i in enumerate(range(N - 1, -1, -1)):
            ch = channels[i]

            # UpSample for every level except the deepest (j=0)
            if j > 0:
                # channels[i+1] is the channel count coming from the deeper level
                self.dec_ups.append(UpSample(channels[i + 1]))

            # 1×1 conv that folds the skip connection back to ch channels
            self.dec_skip_projs.append(nn.Conv2d(2 * ch, ch, 1, bias=False))

            self.dec_blocks.append(
                nn.ModuleList([
                    NAFBlock(ch, drop_out=dropout, time_emb_dim=time_emb_dim)
                    for _ in range(num_dec_nafblocks[i])
                ])
            )

        # ----------------------------------------------------------------
        # Output projection
        # ----------------------------------------------------------------
        self.output_proj = nn.Conv2d(channels[0], in_channels_optical, 1, bias=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Bilinear-resize src to match ref's H×W when they differ by 1 px."""
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:],
                                mode="bilinear", align_corners=False)
        return src

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t:           torch.Tensor,
        t:             torch.Tensor,
        x_cloudy_mean: torch.Tensor,
        sar:           torch.Tensor,
        cloud_mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict clean image x_0 from noised state x_t.

        Args:
            x_t:           Noisy / intermediate diffusion state (B, C_opt, H, W).
            t:             Diffusion timestep  (B,), values in [0, 1].
            x_cloudy_mean: Mean of cloudy observations  (B, C_opt, H, W).
            sar:           SAR image  (B, C_sar, H, W).
            cloud_mask:    Optional cloud binary mask  (B, 1, H, W).
                           Unused in the forward pass; accepted for API
                           compatibility with the bridge training loop.

        Returns:
            Predicted clean image  (B, C_opt, H, W).
        """
        t_emb = self.time_emb(t)      # (B, time_emb_dim)

        # ---- SAR encoder ------------------------------------------------
        # Only run when SFBlock fusion is active; other modes ignore sar_feats.
        sar_feats: List[torch.Tensor] = []
        if self.fusion_mode == "sfblock":
            sar_aligned = self._align(sar, x_t)
            s = self.sar_input(sar_aligned)
            for i in range(self.num_levels):
                for blk in self.sar_enc_blocks[i]:
                    s = blk(s)
                sar_feats.append(s)            # F_sar_i  at scale i
                if i < self.num_levels - 1:
                    s = self.sar_downs[i](s)

        # ---- Optical encoder --------------------------------------------
        # fusion_mode="none": optical-only input (no SAR channels)
        # fusion_mode="sfblock"/"early_concat": SAR concatenated to input
        if self.fusion_mode == "none":
            h = self.opt_input(torch.cat([x_t, x_cloudy_mean], dim=1))
        else:
            sar_aligned = self._align(sar, x_t)
            h = self.opt_input(torch.cat([x_t, x_cloudy_mean, sar_aligned], dim=1))

        enc_skips: List[torch.Tensor] = []

        for i in range(self.num_levels):
            for blk in self.enc_blocks[i]:
                h = blk(h, t_emb)
            # SFBlock fuses SAR encoder features; identity for other modes
            if self.fusion_mode == "sfblock":
                h = self.enc_sfblocks[i](h, sar_feats[i])
            enc_skips.append(h)            # skip_i  after optional SAR fusion
            if i < self.num_levels - 1:
                h = self.enc_downs[i](h)

        # ---- Bottleneck -------------------------------------------------
        for blk in self.bottleneck_blocks:
            h = blk(h)
        h = self.middle_naf(h, t_emb)

        # ---- Decoder ----------------------------------------------------
        #  j iterates from the deepest decoder level to the shallowest.
        #  dec_ups[k] is used at j = k+1.
        up_idx = 0
        for j, i in enumerate(range(self.num_levels - 1, -1, -1)):
            if j > 0:                               # upsample before all but the first
                h = self.dec_ups[up_idx](h)
                up_idx += 1

            skip = enc_skips[i]
            h    = self._align(h, skip)             # guard 1-px padding mismatches
            h    = torch.cat([h, skip], dim=1)      # (B, 2·channels[i], ·)
            h    = self.dec_skip_projs[j](h)        # (B, channels[i], ·)
            for blk in self.dec_blocks[j]:
                h = blk(h, t_emb)

        # ---- Output projection ------------------------------------------
        return self.output_proj(h)                  # (B, C_opt, H, W)


# ---------------------------------------------------------------------------
# Backward-compat alias (previous code imported UNet)
# ---------------------------------------------------------------------------
UNet = SAROpticalUNet


# ---------------------------------------------------------------------------
# Smoke-test  (run as  python -m models.backbone.unet)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    C_opt, C_sar = 4, 2
    B, H, W     = 2, 128, 128   # use 128 to keep the test fast

    model = SAROpticalUNet(
        in_channels_optical = C_opt,
        in_channels_sar     = C_sar,
        base_channels       = 32,          # smaller width for the test
        channel_mult        = [1, 2, 4, 8],
        num_nafblocks       = [1, 1, 1, 4],  # fewer deep blocks for speed
        num_dec_nafblocks   = [1, 1, 1, 1],
        num_vim_blocks      = 1,
        num_heads_sfblock   = [1, 1, 2, 4],
        time_emb_dim        = 64,
    ).to(device)

    # Parameter count
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nSAROpticalUNet  (base_ch=32, mult=[1,2,4,8], naf=[1,1,1,4])")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")

    # Forward pass
    x_t    = torch.randn(B, C_opt, H, W, device=device)
    x_cld  = torch.randn(B, C_opt, H, W, device=device)
    sar    = torch.randn(B, C_sar, H, W, device=device)
    t      = torch.rand(B, device=device)
    mask   = torch.randint(0, 2, (B, 1, H, W), device=device).float()

    out = model(x_t, t, x_cld, sar, mask)

    assert out.shape == (B, C_opt, H, W), f"Unexpected output shape: {out.shape}"
    assert not torch.isnan(out).any(),    "NaN in output!"
    assert not torch.isinf(out).any(),    "Inf in output!"

    print(f"\n  Input : x_t={tuple(x_t.shape)}  t={tuple(t.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"\n  PASSED  ✓")

    # Confirm time conditioning reaches the output
    model.eval()
    with torch.no_grad():
        out_t0 = model(x_t, torch.zeros(B, device=device), x_cld, sar)
        out_t1 = model(x_t, torch.ones(B,  device=device), x_cld, sar)
    assert not torch.allclose(out_t0, out_t1), "t=0 and t=1 give identical outputs — time conditioning may be broken"
    print(f"  Time conditioning check  PASSED  ✓")

    # Confirm SAR conditioning reaches the output
    sar_zeros = torch.zeros_like(sar)
    with torch.no_grad():
        out_sar  = model(x_t, t, x_cld, sar)
        out_nosar = model(x_t, t, x_cld, sar_zeros)
    assert not torch.allclose(out_sar, out_nosar), "SAR input has no effect — SFBlock may be bypassed"
    print(f"  SAR conditioning check   PASSED  ✓\n")
