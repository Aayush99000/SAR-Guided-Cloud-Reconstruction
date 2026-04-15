"""SAR Fusion Block — multi-head cross-modal channel attention.

Channel attention (DB-CR style) fuses SAR features into the optical branch.
Instead of standard spatial attention O(H²W²), it computes attention in the
channel space O(C²), making it practical for large feature maps.

Key idea
--------
For each head with c = C // num_heads channels and L = H*W spatial tokens:

    Q (optical), K, V (SAR)  all shaped  (B, H, c, L)

    A  =  softmax( Q  @  Kᵀ  / √c )      # (B, H, c, c)  — channel affinity
    out  =  A  @  V                        # (B, H, c, L)  — attended features

Row i of A captures how optical channel i attends to all SAR channels.
Complexity: O(B · H · c² · L) instead of O(B · H · L²) for spatial attention.

Public classes
--------------
SFBlock           — channel attention block: SFBlock(channels, num_heads=4).
SARFusionBlock    — backward-compat wrapper used by unet.py;
                    accepts (d_optical, d_sar) with an automatic SAR projection
                    when the channel counts differ.

Reference: DB-CR, Ebel et al. (2022); channel attention formulation from
XCIT (El-Nouby et al., NeurIPS 2021).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SFBlock — multi-head cross-modal channel attention
# ---------------------------------------------------------------------------

class SFBlock(nn.Module):
    """Cross-modal SAR→Optical fusion via multi-head channel attention.

    Args:
        channels:   Feature channel count C (same for optical and SAR inputs).
        num_heads:  Number of attention heads H.  C must be divisible by H.
        ffn_expand: Channel expansion factor for the SimpleGate FFN (default 2).
        dropout:    Dropout on attention weights and FFN output (default 0).
    """

    def __init__(
        self,
        channels:   int,
        num_heads:  int = 4,
        ffn_expand: int = 2,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.channels  = channels
        self.num_heads = num_heads
        self.head_c    = channels // num_heads        # c per head
        self.scale     = self.head_c ** -0.5          # 1 / √c

        # --- Pre-norm ---
        # LayerNorm normalises over the channel dimension.
        # Applied in channel-last format via _ln2d().
        self.norm_opt = nn.LayerNorm(channels)
        self.norm_sar = nn.LayerNorm(channels)

        # --- 1×1 conv projections ---
        # Using Conv2d(1×1) instead of Linear keeps the data in BCHW format and
        # avoids reshape noise; bias=False follows NAFNet / Mamba convention.
        self.q_proj   = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj   = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj   = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # --- SimpleGate FFN (pre-norm) ---
        # conv1 expands to 2×ffn_hidden; SimpleGate halves back to ffn_hidden.
        # conv2 compresses to channels.
        self.norm_ffn  = nn.LayerNorm(channels)
        ffn_hidden     = channels * ffn_expand
        self.ffn_conv1 = nn.Conv2d(channels,    ffn_hidden * 2, 1, bias=True)
        self.ffn_conv2 = nn.Conv2d(ffn_hidden,  channels,       1, bias=True)
        self.ffn_drop  = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ln2d(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm to a (B, C, H, W) tensor (norm over C)."""
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _channel_attention(
        self,
        optical: torch.Tensor,   # (B, C, H, W) — pre-normed
        sar:     torch.Tensor,   # (B, C, H, W) — pre-normed
    ) -> torch.Tensor:           # (B, C, H, W)
        B, C, H, W = optical.shape
        L = H * W

        # 1×1 conv projections  →  (B, C, H, W)
        Q = self.q_proj(optical)
        K = self.k_proj(sar)
        V = self.v_proj(sar)

        # Reshape to (B, num_heads, head_c, L)
        Q = Q.reshape(B, self.num_heads, self.head_c, L)
        K = K.reshape(B, self.num_heads, self.head_c, L)
        V = V.reshape(B, self.num_heads, self.head_c, L)

        # Channel-wise attention: Q @ Kᵀ → (B, H, head_c, head_c)
        #   Q:  (B, H, c, L)  ×  Kᵀ: (B, H, L, c)  →  (B, H, c, c)
        # Attention is in channel space, NOT spatial space.
        # Complexity: O(C² · L)  vs  O(C · L²) for spatial attention.
        A = self.attn_drop(
            F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        )                                        # (B, H, c, c)

        # Apply attention to values: A @ V → (B, H, c, L)
        #   A:  (B, H, c, c)  ×  V: (B, H, c, L)  →  (B, H, c, L)
        out = A @ V                              # (B, H, c, L)

        # Merge heads and project back to (B, C, H, W)
        out = out.reshape(B, C, H, W)
        return self.out_proj(out)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """SimpleGate FFN: pre-norm → expand → gate → compress → residual."""
        h = self._ln2d(self.norm_ffn, x)
        h = self.ffn_conv1(h)                    # (B, 2*ffn_hidden, H, W)
        h1, h2 = h.chunk(2, dim=1)              # SimpleGate: X1 * X2
        h = self.ffn_conv2(h1 * h2)             # (B, channels, H, W)
        return self.ffn_drop(h)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        optical: torch.Tensor,
        sar:     torch.Tensor,
    ) -> torch.Tensor:
        """Fuse SAR features into the optical feature map.

        Args:
            optical: (B, C, H, W) optical / diffusion branch features.
            sar:     (B, C, H, W) SAR encoder features.
                     If spatial size differs from optical, bilinearly resampled.

        Returns:
            (B, C, H, W) fused optical features.
        """
        # Align SAR spatial size if needed (e.g. stride mismatch at bottleneck)
        if sar.shape[2:] != optical.shape[2:]:
            sar = F.interpolate(sar, size=optical.shape[2:],
                                mode="bilinear", align_corners=False)

        # --- Cross-modal channel attention (pre-norm, residual) ---
        opt_n = self._ln2d(self.norm_opt, optical)
        sar_n = self._ln2d(self.norm_sar, sar)
        optical = optical + self._channel_attention(opt_n, sar_n)

        # --- SimpleGate FFN (pre-norm, residual) ---
        optical = optical + self._ffn(optical)

        return optical


# ---------------------------------------------------------------------------
# SARFusionBlock — backward-compatible wrapper for models/backbone/unet.py
# ---------------------------------------------------------------------------

class SARFusionBlock(SFBlock):
    """SFBlock wrapper that accepts different optical and SAR channel counts.

    ``unet.py`` constructs ``SARFusionBlock(ch, cond_channels)`` where
    ``ch`` is the optical channel width and ``cond_channels`` is the SAR
    encoder output width.  When they differ, a learned 1×1 conv projects
    the SAR features to ``d_optical`` channels before fusion.

    Args:
        d_optical:  Optical feature channels (output dimension C).
        d_sar:      SAR feature channels (may differ from d_optical).
        num_heads:  Attention heads (must divide d_optical).
        ffn_expand: FFN expansion factor.
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        d_optical:  int,
        d_sar:      int,
        num_heads:  int   = 4,
        ffn_expand: int   = 2,
        dropout:    float = 0.0,
    ) -> None:
        # Ensure num_heads divides d_optical, fall back to 1 if needed
        while d_optical % num_heads != 0 and num_heads > 1:
            num_heads //= 2

        super().__init__(d_optical, num_heads=num_heads,
                         ffn_expand=ffn_expand, dropout=dropout)

        # SAR channel alignment projection (identity when channels match)
        self.sar_align: nn.Module = (
            nn.Conv2d(d_sar, d_optical, 1, bias=False)
            if d_sar != d_optical else nn.Identity()
        )

    def forward(
        self,
        optical: torch.Tensor,
        sar:     torch.Tensor,
    ) -> torch.Tensor:
        """Same interface as SFBlock; projects SAR channels if necessary."""
        return super().forward(optical, self.sar_align(sar))


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 64, 32, 32

    # --- SFBlock (same channels) ---
    block = SFBlock(channels=C, num_heads=4)
    opt   = torch.randn(B, C, H, W)
    sar   = torch.randn(B, C, H, W)
    out   = block(opt, sar)
    assert out.shape == (B, C, H, W), f"SFBlock shape: {out.shape}"
    assert not torch.allclose(out, opt), "SFBlock is a pass-through — check attention"
    params = sum(p.numel() for p in block.parameters())
    print(f"SFBlock           ({B},{C},{H},{W}) → {tuple(out.shape)}  params: {params:,}  PASSED")

    # --- SARFusionBlock (different channels) ---
    d_sar = 128
    fusion = SARFusionBlock(d_optical=C, d_sar=d_sar, num_heads=4)
    sar2   = torch.randn(B, d_sar, H, W)
    out2   = fusion(opt, sar2)
    assert out2.shape == (B, C, H, W), f"SARFusionBlock shape: {out2.shape}"
    params2 = sum(p.numel() for p in fusion.parameters())
    print(f"SARFusionBlock    ({B},{C},{H},{W}) ← SAR({d_sar}) → {tuple(out2.shape)}  params: {params2:,}  PASSED")

    # --- Spatial size mismatch (SAR at half resolution) ---
    sar3 = torch.randn(B, d_sar, H // 2, W // 2)
    out3 = fusion(opt, sar3)
    assert out3.shape == (B, C, H, W)
    print(f"SARFusionBlock    spatial mismatch ({H//2}×{W//2} SAR → {H}×{W} output)  PASSED")

    # --- Verify attention is in channel space (not spatial) ---
    # Attention map A should be (B, num_heads, head_c, head_c)
    # Total ops ∝ C² × L,  not  L² × C
    head_c = C // 4
    L = H * W
    spatial_ops = C * L * L
    channel_ops = C * C * L
    print(f"\nOp-count comparison at C={C}, L={H}×{W}={L}:")
    print(f"  Spatial attention : ∝ {spatial_ops:,}")
    print(f"  Channel attention : ∝ {channel_ops:,}  ({spatial_ops//channel_ops}× cheaper)")
