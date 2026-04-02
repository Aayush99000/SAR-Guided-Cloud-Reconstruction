"""Vision Mamba bidirectional SSM block.

Implements a simplified version of the Vim (Vision Mamba) selective state-space
model with bidirectional scanning for image feature extraction.

Reference:
  Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with
  Bidirectional State Space Model", arXiv 2401.13587.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Selective Scan (S6) — simplified einsum implementation
# (production code should use the CUDA kernel from mamba-ssm package)
# ---------------------------------------------------------------------------

def selective_scan(
    u: torch.Tensor,      # (B, L, D)
    delta: torch.Tensor,  # (B, L, D)
    A: torch.Tensor,      # (D, N) — log parameterisation
    B: torch.Tensor,      # (B, L, N)
    C: torch.Tensor,      # (B, L, N)
    D: torch.Tensor,      # (D,)
) -> torch.Tensor:
    """Sequential selective scan (reference implementation, not optimised).

    Returns y of shape (B, L, D).
    """
    B_batch, L, d_inner = u.shape
    N = A.shape[1]

    # Discretise A and B via ZOH
    dA = torch.exp(
        torch.einsum("bld,dn->bldn", delta, A)
    )                         # (B, L, D, N)
    dB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

    # Scan
    h = torch.zeros(B_batch, d_inner, N, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(L):
        h = dA[:, i] * h + dB_u[:, i]             # (B, D, N)
        y = torch.einsum("bdn,bn->bd", h, C[:, i]) # (B, D)
        ys.append(y)

    y = torch.stack(ys, dim=1)                     # (B, L, D)
    y = y + u * D                                   # skip connection
    return y


# ---------------------------------------------------------------------------
# SSM core
# ---------------------------------------------------------------------------

class VimSSM(nn.Module):
    """Single-direction Selective State Space Model layer.

    Args:
        d_model:  Input / output feature dimension.
        d_state:  SSM state dimension N.
        d_conv:   Local depthwise conv kernel size.
        expand:   Inner expansion factor.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner = int(d_model * expand)

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)  # B, C, delta
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape
        xz = self.in_proj(x)                             # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                   # each (B, L, d_inner)

        # Local conv
        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = self.act(x_conv)

        # SSM parameters
        x_dbl = self.x_proj(x_conv)                     # (B, L, N*2+d_inner)
        delta_raw = x_dbl[..., : self.d_inner]
        B_ssm = x_dbl[..., self.d_inner : self.d_inner + self.d_state]
        C_ssm = x_dbl[..., self.d_inner + self.d_state :]

        delta = F.softplus(self.dt_proj(delta_raw))     # (B, L, d_inner)
        A = -torch.exp(self.A_log)                      # (d_inner, N)

        y = selective_scan(x_conv, delta, A, B_ssm, C_ssm, self.D)
        y = y * self.act(z)

        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Bidirectional Mamba block
# ---------------------------------------------------------------------------

class BidirectionalMamba(nn.Module):
    """Vim-style bidirectional SSM block with residual connection.

    Scans the sequence in both forward and backward directions and merges results.

    Args:
        d_model:     Feature dimension.
        d_state:     SSM state size.
        d_conv:      Local conv kernel size.
        expand:      SSM inner expansion ratio.
        dropout:     Dropout on the output projection.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd_ssm = VimSSM(d_model, d_state, d_conv, expand)
        self.bwd_ssm = VimSSM(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(d_model * 2, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) token sequence.

        Returns:
            (B, L, d_model)
        """
        h = self.norm(x)
        fwd = self.fwd_ssm(h)
        bwd = self.bwd_ssm(torch.flip(h, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])

        out = self.merge(torch.cat([fwd, bwd], dim=-1))
        return x + self.drop(out)
