"""SAR Fusion Block (SFBlock) — cross-attention between optical and SAR features.

The SFBlock allows the optical reconstruction branch to query SAR features as
keys and values, injecting radar texture and structural information into the
optical latent stream at each decoder resolution.

Architecture:
    Optical query: LayerNorm → project to Q
    SAR key/value: LayerNorm → project to K, V
    Scaled dot-product cross-attention → output projection
    Residual + FFN (with SimpleGate activation)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SARFusionBlock(nn.Module):
    """Cross-attention SAR ↔ Optical fusion block.

    Args:
        d_optical:    Optical feature channels (query dimension).
        d_sar:        SAR feature channels (key/value dimension).
        num_heads:    Number of attention heads.
        ffn_expand:   FFN hidden dimension multiplier.
        dropout:      Attention / FFN dropout probability.
        bias:         Whether to use bias in projection layers.
    """

    def __init__(
        self,
        d_optical: int = 256,
        d_sar: int = 256,
        num_heads: int = 8,
        ffn_expand: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_optical % num_heads == 0, "d_optical must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_optical // num_heads
        self.scale = self.head_dim ** -0.5

        # Norms
        self.norm_opt = nn.LayerNorm(d_optical)
        self.norm_sar = nn.LayerNorm(d_sar)
        self.norm_ffn = nn.LayerNorm(d_optical)

        # Cross-attention projections
        self.q_proj = nn.Linear(d_optical, d_optical, bias=bias)
        self.k_proj = nn.Linear(d_sar, d_optical, bias=bias)
        self.v_proj = nn.Linear(d_sar, d_optical, bias=bias)
        self.out_proj = nn.Linear(d_optical, d_optical, bias=bias)

        self.attn_drop = nn.Dropout(dropout)

        # FFN with SimpleGate activation
        ffn_hidden = d_optical * ffn_expand
        self.ffn_fc1 = nn.Linear(d_optical, ffn_hidden * 2)  # *2 for SimpleGate split
        self.ffn_fc2 = nn.Linear(ffn_hidden, d_optical)
        self.ffn_drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """(B, C, H, W) → (B, H*W, C), returns original spatial shape."""
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C), (H, W)

    def _to_spatial(self, x: torch.Tensor, hw: tuple) -> torch.Tensor:
        """(B, H*W, C) → (B, C, H, W)."""
        B, L, C = x.shape
        H, W = hw
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def _cross_attention(
        self,
        q_tokens: torch.Tensor,   # (B, Lq, C)
        k_tokens: torch.Tensor,   # (B, Lk, C)
        v_tokens: torch.Tensor,   # (B, Lk, C)
    ) -> torch.Tensor:
        B, Lq, C = q_tokens.shape
        Lk = k_tokens.shape[1]

        q = self.q_proj(q_tokens).reshape(B, Lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k_tokens).reshape(B, Lk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v_tokens).reshape(B, Lk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, Lq, C)
        return self.out_proj(out)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_ffn(x)
        h = self.ffn_fc1(h)
        x1, x2 = h.chunk(2, dim=-1)   # SimpleGate
        h = self.ffn_fc2(x1 * x2)
        return self.ffn_drop(h)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        sar_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse SAR features into the optical feature map.

        Args:
            optical:   Optical feature map (B, d_optical, H, W).
            sar:       SAR feature map (B, d_sar, H', W').
                       H', W' need not equal H, W — will be adaptively pooled.
            sar_mask:  Optional cloud mask (B, 1, H, W) — not used in attention
                       itself but reserved for future mask-guided attention.

        Returns:
            Fused optical feature map (B, d_optical, H, W).
        """
        # Align SAR spatial size to optical if needed
        if sar.shape[2:] != optical.shape[2:]:
            sar = F.interpolate(sar, size=optical.shape[2:], mode="bilinear", align_corners=False)

        opt_tok, hw = self._to_tokens(optical)
        sar_tok, _ = self._to_tokens(sar)

        # Cross-attention: optical queries SAR keys/values
        q = self.norm_opt(opt_tok)
        kv = self.norm_sar(sar_tok)
        attn_out = self._cross_attention(q, kv, kv)
        opt_tok = opt_tok + attn_out

        # FFN
        opt_tok = opt_tok + self._ffn(opt_tok)

        return self._to_spatial(opt_tok, hw)
