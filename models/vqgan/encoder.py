"""VQ-GAN Encoder: maps (B, C_in, H, W) → quantised latent (B, latent_dim, H/f, W/f).

Down-sampling factor f = 2^(len(channels)-1).  A vector-quantisation bottleneck
is embedded so this module can be used standalone or as part of a full VQ-GAN.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttnBlock(nn.Module):
    """Single-head self-attention (spatial)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv.unbind(dim=1)                          # each (B, C, HW)
        attn = torch.softmax(
            torch.bmm(q.permute(0, 2, 1), k) * self.scale, dim=-1
        )                                                     # (B, HW, HW)
        h = torch.bmm(v, attn.permute(0, 2, 1))             # (B, C, HW)
        h = self.proj(h.reshape(B, C, H, W))
        return x + h


# ---------------------------------------------------------------------------
# Vector Quantiser
# ---------------------------------------------------------------------------

class VectorQuantiser(nn.Module):
    """Straight-through VQ bottleneck.

    Args:
        num_embeddings:  Codebook size K.
        embedding_dim:   Dimension of each code vector.
        commitment_cost: β in the commitment loss term.
    """

    def __init__(
        self,
        num_embeddings: int = 8192,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = commitment_cost

        self.codebook = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.K, 1.0 / self.K)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D, H, W) continuous latent.

        Returns:
            z_q:          Quantised latent (B, D, H, W).
            indices:      Codebook indices (B, H*W).
            commit_loss:  Scalar commitment loss.
        """
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)               # (BHW, D)

        # Distances ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z^T*e
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            + self.codebook.weight.pow(2).sum(1)
            - 2.0 * z_flat @ self.codebook.weight.t()
        )                                                            # (BHW, K)
        indices = dist.argmin(dim=1)                                 # (BHW,)
        z_q = self.codebook(indices).reshape(B, H, W, D).permute(0, 3, 1, 2)

        commit_loss = F.mse_loss(z_q.detach(), z) * self.beta
        z_q = z + (z_q - z).detach()                                # straight-through

        return z_q, indices.reshape(B, H * W), commit_loss


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VQGANEncoder(nn.Module):
    """Hierarchical CNN encoder with a VQ bottleneck.

    Args:
        in_channels:     Input image channels.
        channels:        Feature channel sizes at each resolution level.
        latent_dim:      Output latent dimension (= codebook embedding_dim).
        num_res_blocks:  Residual blocks per resolution.
        num_embeddings:  VQ codebook size.
        commitment_cost: VQ commitment loss weight.
        attn_resolutions: Spatial sizes at which to add self-attention.
    """

    def __init__(
        self,
        in_channels: int = 13,
        channels: List[int] = None,
        latent_dim: int = 256,
        num_res_blocks: int = 2,
        num_embeddings: int = 8192,
        commitment_cost: float = 0.25,
        attn_resolutions: List[int] = None,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        attn_resolutions = attn_resolutions or [16]

        layers: List[nn.Module] = [nn.Conv2d(in_channels, channels[0], 3, padding=1)]

        current_res = 256   # assumed input spatial size
        in_ch = channels[0]
        for out_ch in channels:
            for _ in range(num_res_blocks):
                layers.append(ResBlock(in_ch))
                if current_res in attn_resolutions:
                    layers.append(AttnBlock(in_ch))
            # Down-sample
            layers.append(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch
            current_res //= 2

        layers += [
            ResBlock(in_ch),
            AttnBlock(in_ch),
            ResBlock(in_ch),
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, latent_dim, 3, padding=1),
        ]
        self.encoder = nn.Sequential(*layers)
        self.vq = VectorQuantiser(num_embeddings, latent_dim, commitment_cost)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_q:          Quantised latent.
            indices:      Flat codebook indices.
            commit_loss:  VQ commitment loss.
        """
        z = self.encoder(x)
        z_q, indices, commit_loss = self.vq(z)
        return z_q, indices, commit_loss
