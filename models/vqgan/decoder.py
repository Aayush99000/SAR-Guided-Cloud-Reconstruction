"""VQ-GAN Decoder: maps quantised latent (B, latent_dim, H/f, W/f) → image (B, C_out, H, W)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .encoder import ResBlock, AttnBlock


class VQGANDecoder(nn.Module):
    """Mirror of the VQGANEncoder with bilinear up-sampling instead of strided conv.

    Args:
        out_channels:    Output image channels (should match encoder in_channels).
        channels:        Feature channel sizes in *decode* order (coarse → fine).
                         Typically the reverse of the encoder channel list.
        latent_dim:      Input latent dimension.
        num_res_blocks:  Residual blocks per resolution.
        attn_resolutions: Spatial sizes at which to add self-attention.
    """

    def __init__(
        self,
        out_channels: int = 13,
        channels: List[int] = None,
        latent_dim: int = 256,
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = None,
    ) -> None:
        super().__init__()
        channels = channels or [512, 256, 128, 64]
        attn_resolutions = attn_resolutions or [16]

        in_ch = channels[0]
        layers: List[nn.Module] = [
            nn.Conv2d(latent_dim, in_ch, 3, padding=1),
            ResBlock(in_ch),
            AttnBlock(in_ch),
            ResBlock(in_ch),
        ]

        current_res = 16    # coarsest spatial size after encoder
        for out_ch in channels:
            for _ in range(num_res_blocks):
                layers.append(ResBlock(in_ch))
                if current_res in attn_resolutions:
                    layers.append(AttnBlock(in_ch))
            # Up-sample
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
            ]
            in_ch = out_ch
            current_res *= 2

        layers += [
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
            nn.Tanh(),   # output in [-1, 1]
        ]
        self.decoder = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: (B, latent_dim, H/f, W/f) quantised latent.

        Returns:
            Reconstructed image (B, out_channels, H, W).
        """
        return self.decoder(z_q)
