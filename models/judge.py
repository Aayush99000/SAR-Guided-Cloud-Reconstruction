"""Frozen land-cover judge: a U-Net with an ImageNet-pretrained timm encoder.

Trained once on real, clear Sentinel-2 images against ESA WorldCover labels
(``train/train_judge.py``), then frozen and applied to reconstructions
(``train/evaluate_judge.py``).  It never sees bridge outputs during training.

Input: the bridge's optical bands in dataset order and normalisation
(B2, B3, B4, B8 reflectance / 10000).  The first three are reordered to RGB so
the pretrained stem's colour filters line up; per-band standardisation uses
statistics estimated from the training set and stored in the checkpoint.
"""

from __future__ import annotations

from typing import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.body(x)


class LandCoverJudge(nn.Module):
    """U-Net segmentation model over the bridge's optical bands.

    Args:
        num_classes:  Output classes.
        in_channels:  Optical bands (4 for B2, B3, B4, B8).
        rgb_order:    Indices of R, G, B among the input bands, moved to the
                      front before the encoder (default B4, B3, B2).
        encoder:      timm backbone name.
        pretrained:   Load ImageNet weights for the encoder.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 4,
        rgb_order: Sequence[int] = (2, 1, 0),
        encoder: str = "resnet34",
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()
        rest = [i for i in range(in_channels) if i not in rgb_order]
        self.register_buffer("band_order", torch.tensor(list(rgb_order) + rest), persistent=False)
        self.register_buffer("mean", torch.zeros(1, in_channels, 1, 1))
        self.register_buffer("std", torch.ones(1, in_channels, 1, 1))
        self.config = dict(num_classes=num_classes, in_channels=in_channels, rgb_order=list(rgb_order),
                           encoder=encoder, decoder_channels=list(decoder_channels))

        self.encoder = timm.create_model(encoder, features_only=True, pretrained=pretrained, in_chans=in_channels)
        enc_ch = self.encoder.feature_info.channels()          # strides 2, 4, 8, 16, 32
        skips = list(reversed(enc_ch[:-1])) + [0]
        blocks, ch = [], enc_ch[-1]
        for skip_ch, out_ch in zip(skips, decoder_channels):
            blocks.append(_DecoderBlock(ch, skip_ch, out_ch))
            ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(ch, num_classes, 1)

    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Per-band statistics in input (dataset) band order."""
        self.mean.copy_(mean.view_as(self.mean))
        self.std.copy_(std.clamp_min(1e-6).view_as(self.std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ((x - self.mean) / self.std)[:, self.band_order]
        feats = self.encoder(x)
        y = feats[-1]
        for block, skip in zip(self.decoder, list(reversed(feats[:-1])) + [None]):
            y = block(y, skip)
        return self.head(y)

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu") -> "LandCoverJudge":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"], pretrained=False)
        model.load_state_dict(ckpt["model"])
        return model.eval()
