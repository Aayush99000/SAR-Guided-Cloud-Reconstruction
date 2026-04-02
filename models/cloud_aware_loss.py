"""Adaptive cloud-weighted reconstruction loss.

The loss up-weights pixels in cloud-covered regions so the network focuses on
reconstructing the hardest (and most useful) areas.

Components:
  - MSE  (L2) loss                    — scaled by lambda_mse
  - SSIM loss  (1 - SSIM)             — scaled by lambda_ssim
  - Perceptual loss (VGG-16 features) — scaled by lambda_perceptual (optional)
  - Adversarial loss                  — scaled by lambda_adversarial (optional)

Cloud weighting:
  Pixel-wise weight w(x,y) = alpha + (1 - alpha) * mask(x,y)
  where alpha ∈ (0, 1) and mask = 1 in cloudy regions, 0 elsewhere.
  Setting alpha=1 recovers uniform weighting; alpha≈0 focuses on cloud pixels.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM helper (2-D, single-scale)
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable SSIM loss: 1 - SSIM(pred, target).

    Operates channel-by-channel on (B, C, H, W) tensors in [-1, 1].
    """
    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size).to(pred.device)
    kernel = kernel.expand(C, 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)

    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu12

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    ssim_map = (
        (2 * mu12 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    loss = 1.0 - ssim_map
    if reduction == "mean":
        return loss.mean()
    return loss  # (B, C, H, W) for per-pixel weighting


# ---------------------------------------------------------------------------
# Cloud-aware loss module
# ---------------------------------------------------------------------------

class CloudAwareLoss(nn.Module):
    """Adaptive cloud-weighted multi-component reconstruction loss.

    Args:
        cloud_loss_alpha:   Base weight for clear pixels (α).  Cloud pixels get
                            weight 1.0; clear pixels get weight α. Default 0.8.
        lambda_mse:         Weight for pixel-wise MSE term.
        lambda_ssim:        Weight for SSIM term.
        lambda_perceptual:  Weight for VGG perceptual loss (0 to disable).
        lambda_adversarial: Weight for adversarial loss (0 to disable).
    """

    def __init__(
        self,
        cloud_loss_alpha: float = 0.8,
        lambda_mse: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_perceptual: float = 0.1,
        lambda_adversarial: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = cloud_loss_alpha
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adversarial = lambda_adversarial

    def _pixel_weights(self, mask: Optional[torch.Tensor], shape: torch.Size) -> torch.Tensor:
        """Build per-pixel loss weight map.

        Args:
            mask:  (B, 1, H, W) binary cloud mask (1 = cloud, 0 = clear), or None.
            shape: (B, C, H, W) of the prediction.

        Returns:
            Weight tensor (B, 1, H, W).
        """
        if mask is None:
            return torch.ones(shape[0], 1, shape[2], shape[3], device=shape)

        # Cloud pixels → weight 1.0; clear pixels → weight alpha
        weights = self.alpha + (1.0 - self.alpha) * mask
        return weights  # (B, 1, H, W)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cloud_mask: Optional[torch.Tensor] = None,
        discriminator_logits: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the cloud-aware loss.

        Args:
            pred:                   Reconstructed image (B, C, H, W) in [-1, 1].
            target:                 Ground-truth clear image (B, C, H, W) in [-1, 1].
            cloud_mask:             Optional binary cloud mask (B, 1, H, W).
            discriminator_logits:   Optional real/fake logits for adversarial term.

        Returns:
            total_loss: Scalar loss tensor.
            breakdown:  Dict with individual loss components for logging.
        """
        # --- Pixel weights ---
        w = self._pixel_weights(cloud_mask, pred.shape)  # (B, 1, H, W)

        # --- Weighted MSE ---
        mse = ((pred - target) ** 2 * w).mean()

        # --- Weighted SSIM ---
        ssim_map = ssim_loss(pred, target, reduction="none")   # (B, C, H, W)
        ssim = (ssim_map * w).mean()

        total = self.lambda_mse * mse + self.lambda_ssim * ssim
        breakdown = {
            "mse": mse.item(),
            "ssim": ssim.item(),
        }

        # --- Perceptual loss (VGG) ---
        if self.lambda_perceptual > 0:
            perceptual = self._perceptual_loss(pred, target)
            total = total + self.lambda_perceptual * perceptual
            breakdown["perceptual"] = perceptual.item()

        # --- Adversarial loss ---
        if self.lambda_adversarial > 0 and discriminator_logits is not None:
            adv_loss = F.binary_cross_entropy_with_logits(
                discriminator_logits,
                torch.ones_like(discriminator_logits),
            )
            total = total + self.lambda_adversarial * adv_loss
            breakdown["adversarial"] = adv_loss.item()

        breakdown["total"] = total.item()
        return total, breakdown

    # ------------------------------------------------------------------

    def _perceptual_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Stub: replace with VGG feature-matching loss for full implementation.

        Operates on the first 3 channels (RGB-equivalent) of the input.
        """
        # Minimal L1 on pixel space as placeholder; swap in VGG features when available
        return F.l1_loss(pred[:, :3], target[:, :3])
