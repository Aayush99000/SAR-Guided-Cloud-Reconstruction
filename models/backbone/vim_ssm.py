"""Vision Mamba bidirectional SSM blocks.

Two public classes:

  VimBlock          — 2D-native block (B,C,H,W) → (B,C,H,W).
                      Flattens spatially, adds learnable position embeddings,
                      runs bidirectional selective scan, reshapes back.
                      Use this in new code.

  BidirectionalMamba — sequence-level block (B,L,D) → (B,L,D).
                       Kept for backward compatibility with models/backbone/unet.py
                       which handles the 2D↔sequence conversion itself.

SSM kernel priority
-------------------
1. ``mamba_ssm.ops.selective_scan_interface.selective_scan_fn``  (CUDA, fast)
2. Pure-PyTorch reference implementation                          (slow fallback)

Reference:
  Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with
  Bidirectional State Space Model", ICML 2024.  arXiv:2401.13587
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSM kernel — CUDA fast path or pure-PyTorch fallback
# ---------------------------------------------------------------------------

_selective_scan_fn: Optional[Callable] = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
    _selective_scan_fn = _cuda_scan
except Exception:
    pass   # fall through to reference implementation


def _selective_scan_ref(
    u: torch.Tensor,      # (B, D, L)
    delta: torch.Tensor,  # (B, D, L)
    A: torch.Tensor,      # (D, N)  — already negated (A < 0)
    B: torch.Tensor,      # (B, N, L)
    C: torch.Tensor,      # (B, N, L)
    D: torch.Tensor,      # (D,)
) -> torch.Tensor:
    """Sequential selective scan — reference implementation.

    Zero-order-hold discretisation:
        ΔA_bar[t] = exp(delta[t] ⊗ A)           shape (B, D, N)
        ΔB_bar[t] = delta[t, :, None] * B[t]    shape (B, D, N)
        h[t]      = ΔA_bar[t] * h[t-1] + ΔB_bar[t] * u[t, :, None]
        y[t]      = sum_n( h[t] * C[t] ) + D * u[t]

    All tensors in (B, D, L) layout  →  returns (B, D, L).
    """
    B_batch, D_dim, L = u.shape
    N = A.shape[1]
    dtype = u.dtype

    # Pre-compute discretised matrices for all time steps at once to
    # avoid Python-loop overhead on the einsum (still O(L) memory).
    # delta: (B, D, L) → (B, D, L, 1)  ×  A: (D, N) → (1, D, 1, N)
    dA = torch.exp(delta.unsqueeze(-1) * A[None, :, None, :])  # (B, D, L, N)
    # dB * u:  delta (B,D,L) × B (B,N,L) × u (B,D,L) → (B, D, L, N)
    dBu = (delta * u).unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1)  # (B,D,L,N)

    h = torch.zeros(B_batch, D_dim, N, device=u.device, dtype=dtype)
    ys: list[torch.Tensor] = []
    for t in range(L):
        h = dA[:, :, t, :] * h + dBu[:, :, t, :]       # (B, D, N)
        # y[t] = C[t] · h[t]  — C: (B, N, L)
        y_t = (h * C[:, :, t].unsqueeze(1)).sum(-1)     # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=2)                           # (B, D, L)
    y = y + D[None, :, None] * u                        # skip connection
    return y


def _run_scan(
    u: torch.Tensor,      # (B, D, L) — conv-activated input
    delta: torch.Tensor,  # (B, D, L) — after softplus
    A: torch.Tensor,      # (D, N)
    B: torch.Tensor,      # (B, N, L)
    C: torch.Tensor,      # (B, N, L)
    D: torch.Tensor,      # (D,)
) -> torch.Tensor:        # (B, D, L)
    """Dispatch to CUDA kernel or reference implementation.

    The CUDA kernel requires all tensors to be on a CUDA device.
    Falls back to the pure-PyTorch implementation on CPU.
    """
    if _selective_scan_fn is not None and u.is_cuda:
        return _selective_scan_fn(
            u.float(), delta.float(), A.float(),
            B.float(), C.float(), D.float(),
            z=None, delta_bias=None, delta_softplus=False,
        ).to(u.dtype)
    return _selective_scan_ref(u, delta, A, B, C, D)


# ---------------------------------------------------------------------------
# Single-direction SSM helper
# ---------------------------------------------------------------------------

def _ssm_one_direction(
    x_in: torch.Tensor,       # (B, L, d_inner)
    conv1d: nn.Conv1d,
    x_proj: nn.Linear,        # d_inner → dt_rank + 2*d_state
    dt_proj: nn.Linear,       # dt_rank → d_inner  (has bias)
    A_log: nn.Parameter,      # (d_inner, d_state)
    D_param: nn.Parameter,    # (d_inner,)
    dt_rank: int,
    d_state: int,
) -> torch.Tensor:             # (B, L, d_inner)
    """One forward pass of the selective scan (one direction)."""
    B, L, d_inner = x_in.shape

    # Depthwise causal conv (trim causal padding)
    x_c = conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)  # (B, L, d_inner)
    x_c = F.silu(x_c)

    # Data-dependent SSM parameters
    x_dbl = x_proj(x_c)                                            # (B, L, dt_rank+2N)
    delta_raw = x_dbl[..., :dt_rank]                               # (B, L, dt_rank)
    B_ssm    = x_dbl[..., dt_rank : dt_rank + d_state]            # (B, L, N)
    C_ssm    = x_dbl[..., dt_rank + d_state:]                     # (B, L, N)

    delta = F.softplus(dt_proj(delta_raw))                         # (B, L, d_inner)
    A = -torch.exp(A_log.float())                                  # (d_inner, N)

    y = _run_scan(
        x_c.transpose(1, 2).contiguous(),      # (B, d_inner, L)
        delta.transpose(1, 2).contiguous(),     # (B, d_inner, L)
        A,
        B_ssm.transpose(1, 2).contiguous(),    # (B, N, L)
        C_ssm.transpose(1, 2).contiguous(),    # (B, N, L)
        D_param,
    )                                           # (B, d_inner, L)
    return y.transpose(1, 2)                   # (B, L, d_inner)


# ---------------------------------------------------------------------------
# VimBlock — 2D-native bidirectional Mamba block
# ---------------------------------------------------------------------------

class VimBlock(nn.Module):
    """Bidirectional State Space Model block for 2D feature maps.

    Takes (B, C, H, W), flattens spatially to a token sequence, adds learnable
    1D position embeddings, runs forward and backward selective scans, gates with
    a learned z branch, then reshapes back to (B, C, H, W).

    Architecture (one block):
        x_2d  →  flatten → x_seq + pos_emb
                              ↓  LayerNorm
                   in_proj → x_in (E)  |  z (E)
                              ↓
             fwd: Conv1d → SiLU → SSM(A,B,C,delta) → y_fwd
             bwd: flip → Conv1d → SiLU → SSM → flip → y_bwd
                              ↓
             y = (y_fwd + y_bwd) * SiLU(z)
                              ↓
                   out_proj  →  dropout  →  + residual
                              ↓
                         reshape (B, C, H, W)

    Args:
        dim:        Input / output channel count C.
        d_state:    SSM state dimension N (default 16).
        expand:     Inner expansion factor E = expand * dim (default 2).
        dt_rank:    Rank of the Δ projection. ``"auto"`` → ceil(dim/16).
        d_conv:     Depthwise conv kernel size (default 4).
        dropout:    Dropout applied after out_proj (default 0).
        max_seq_len: Maximum supported H*W for the learnable pos embedding.
                    Longer sequences are handled by bilinear interpolation.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        d_conv: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.dim     = dim
        self.d_state = d_state
        self.d_inner = d_inner = int(dim * expand)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else int(dt_rank)

        # --- Learnable 1D position embedding (broadcast over batch) ---
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # --- Shared normalisation + input projection ---
        self.norm    = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)   # → x_in, z

        # --- Forward-direction SSM parameters ---
        self.fwd_conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                    padding=d_conv - 1, groups=d_inner)
        self.fwd_x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.fwd_dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        A_fwd = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.fwd_A_log  = nn.Parameter(torch.log(A_fwd))
        self.fwd_D      = nn.Parameter(torch.ones(d_inner))

        # --- Backward-direction SSM parameters (separate weights) ---
        self.bwd_conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                    padding=d_conv - 1, groups=d_inner)
        self.bwd_x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.bwd_dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        A_bwd = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.bwd_A_log  = nn.Parameter(torch.log(A_bwd))
        self.bwd_D      = nn.Parameter(torch.ones(d_inner))

        # --- Output projection ---
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        self.drop     = nn.Dropout(dropout)

    def _get_pos_emb(self, L: int, device: torch.device) -> torch.Tensor:
        """Return pos embedding of length L, interpolating if necessary."""
        if L == self.pos_emb.shape[1]:
            return self.pos_emb.to(device)
        # Bilinear interpolation along the sequence dimension
        return F.interpolate(
            self.pos_emb.transpose(1, 2),   # (1, dim, L_max)
            size=L, mode="linear", align_corners=False,
        ).transpose(1, 2).to(device)        # (1, L, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map.
        Returns:
            (B, C, H, W) — same shape.
        """
        B, C, H, W = x.shape
        L = H * W

        # 1. Reshape to sequence (B, L, C) and add position embeddings
        x_seq = x.flatten(2).transpose(1, 2)          # (B, L, C)
        x_seq = x_seq + self._get_pos_emb(L, x.device)

        # 2. LayerNorm + project to x_in and z
        h = self.norm(x_seq)
        xz    = self.in_proj(h)                        # (B, L, 2*E)
        x_in, z = xz.chunk(2, dim=-1)                 # each (B, L, E)

        # 3. Forward scan
        y_fwd = _ssm_one_direction(
            x_in,
            self.fwd_conv1d, self.fwd_x_proj, self.fwd_dt_proj,
            self.fwd_A_log,  self.fwd_D,
            self.dt_rank,    self.d_state,
        )                                              # (B, L, E)

        # 4. Backward scan (flip → scan → flip back)
        y_bwd = _ssm_one_direction(
            torch.flip(x_in, dims=[1]),
            self.bwd_conv1d, self.bwd_x_proj, self.bwd_dt_proj,
            self.bwd_A_log,  self.bwd_D,
            self.dt_rank,    self.d_state,
        )
        y_bwd = torch.flip(y_bwd, dims=[1])            # (B, L, E)

        # 5. Gate: combine directions, modulate by z
        y = (y_fwd + y_bwd) * F.silu(z)               # (B, L, E)

        # 6. Project back to dim + residual
        out = self.drop(self.out_proj(y))              # (B, L, C)
        out = x_seq + out                              # residual on sequence

        # 7. Reshape to (B, C, H, W)
        return out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# VimSSM — single-direction SSM layer (sequence in → sequence out)
# Used internally by BidirectionalMamba.
# ---------------------------------------------------------------------------

class VimSSM(nn.Module):
    """Single-direction SSM: (B, L, d_model) → (B, L, d_model).

    Encapsulates: in_proj → conv → SSM → z-gate → out_proj.

    Args:
        d_model: Input / output feature dimension.
        d_state: SSM state size N.
        d_conv:  Depthwise conv kernel size.
        expand:  Inner expansion ratio.
        dt_rank: Δ projection rank (``"auto"`` → ceil(d_model/16)).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
        dt_rank: Union[int, str] = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner = int(d_model * expand)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                  padding=d_conv - 1, groups=d_inner)
        self.act      = nn.SiLU()

        # Δ, B, C projection  (delta-first convention)
        self.x_proj   = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)."""
        B, L, _ = x.shape
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                # each (B, L, d_inner)

        y = _ssm_one_direction(
            x_in,
            self.conv1d, self.x_proj, self.dt_proj,
            self.A_log, self.D,
            self.dt_rank, self.d_state,
        )                                              # (B, L, d_inner)
        y = y * self.act(z)                           # z-gate
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# BidirectionalMamba — sequence-level block kept for unet.py compatibility
# ---------------------------------------------------------------------------

class BidirectionalMamba(nn.Module):
    """Vim-style bidirectional SSM block operating on token sequences.

    Used by models/backbone/unet.py which handles 2D ↔ sequence reshaping
    externally.  For new 2D code use ``VimBlock`` instead.

    Args:
        d_model:  Feature dimension.
        d_state:  SSM state size.
        d_conv:   Depthwise conv kernel size.
        expand:   Inner expansion ratio.
        dt_rank:  Δ rank (``"auto"`` → ceil(d_model/16)).
        dropout:  Output dropout probability.
    """

    def __init__(
        self,
        d_model:  int = 256,
        d_state:  int = 16,
        d_conv:   int = 4,
        expand:   int = 2,
        dt_rank:  Union[int, str] = "auto",
        dropout:  float = 0.0,
    ) -> None:
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.fwd_ssm = VimSSM(d_model, d_state, d_conv, expand, dt_rank)
        self.bwd_ssm = VimSSM(d_model, d_state, d_conv, expand, dt_rank)
        # Merge forward + backward outputs (concat → project)
        self.merge   = nn.Linear(d_model * 2, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)."""
        h   = self.norm(x)
        fwd = self.fwd_ssm(h)
        bwd = self.bwd_ssm(torch.flip(h, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])
        out = self.merge(torch.cat([fwd, bwd], dim=-1))
        return x + self.drop(out)


# ---------------------------------------------------------------------------
# Shape-verification smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kernel = "CUDA" if _selective_scan_fn is not None else "PyTorch fallback"
    print(f"Device: {device}   |   SSM kernel: {kernel}")

    # --- VimBlock (2D-native) ---
    B, C, H, W = 2, 64, 16, 16
    block = VimBlock(dim=C, d_state=16, expand=2, dt_rank="auto").to(device)
    x     = torch.randn(B, C, H, W, device=device)
    out   = block(x)
    assert out.shape == (B, C, H, W), f"VimBlock shape mismatch: {out.shape}"
    print(f"VimBlock          ({B},{C},{H},{W}) → {tuple(out.shape)}  PASSED")

    # Verify residual: output changes from input (non-trivial)
    assert not torch.allclose(out, x), "VimBlock output equals input — trivial pass-through?"
    print("VimBlock non-trivial output  PASSED")

    # --- BidirectionalMamba (sequence-level, backward compat) ---
    L = H * W
    bimamba = BidirectionalMamba(d_model=C, d_state=16).to(device)
    tokens  = torch.randn(B, L, C, device=device)
    out_seq = bimamba(tokens)
    assert out_seq.shape == (B, L, C), f"BidirectionalMamba shape: {out_seq.shape}"
    print(f"BidirectionalMamba({B},{L},{C}) → {tuple(out_seq.shape)}  PASSED")

    # --- Parameter counts ---
    vim_params  = sum(p.numel() for p in block.parameters())
    bimam_params = sum(p.numel() for p in bimamba.parameters())
    print(f"\nParameter counts (C={C}, d_state=16, expand=2):")
    print(f"  VimBlock          : {vim_params:,}")
    print(f"  BidirectionalMamba: {bimam_params:,}")
