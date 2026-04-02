"""Image quality metrics: PSNR, SSIM, MAE, SAM, LPIPS.

All functions operate on (B, C, H, W) torch tensors normalised to [-1, 1].
Metrics are averaged over batch and spatial dimensions unless noted.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 2.0,   # range is [-1, 1] → max_val = 2
) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (dB).

    Args:
        pred, target: (B, C, H, W) in [-1, 1].
        max_val:      Dynamic range of the signal.

    Returns:
        Scalar PSNR averaged over the batch.
    """
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))   # (B,)
    mse = mse.clamp(min=1e-10)
    return (10.0 * torch.log10(max_val ** 2 / mse)).mean()


# ---------------------------------------------------------------------------
# SSIM (single-scale, multi-channel)
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return kernel.expand(channels, 1, size, size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Structural Similarity Index Measure, averaged over batch & channels.

    Returns a value in [0, 1].
    """
    C_img = pred.shape[1]
    kernel = _gaussian_kernel_2d(window_size, channels=C_img).to(pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C_img)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C_img)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1 = F.conv2d(pred ** 2, kernel, padding=pad, groups=C_img) - mu1_sq
    s2 = F.conv2d(target ** 2, kernel, padding=pad, groups=C_img) - mu2_sq
    s12 = F.conv2d(pred * target, kernel, padding=pad, groups=C_img) - mu12

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    )
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error."""
    return (pred - target).abs().mean()


# ---------------------------------------------------------------------------
# SAM (Spectral Angle Mapper)
# ---------------------------------------------------------------------------

def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spectral Angle Mapper (degrees), averaged over batch and pixels.

    Args:
        pred, target: (B, C, H, W) — spectral vectors along the C dimension.

    Returns:
        Mean SAM in degrees.
    """
    dot = (pred * target).sum(dim=1)                         # (B, H, W)
    norm_p = pred.norm(dim=1).clamp(min=eps)
    norm_t = target.norm(dim=1).clamp(min=eps)
    cos_sim = (dot / (norm_p * norm_t)).clamp(-1.0 + eps, 1.0 - eps)
    angle_rad = torch.acos(cos_sim)
    return torch.rad2deg(angle_rad).mean()


# ---------------------------------------------------------------------------
# LPIPS (learned perceptual similarity)
# ---------------------------------------------------------------------------

def lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "vgg",
) -> torch.Tensor:
    """LPIPS using the lpips library.  Falls back to L1 if unavailable."""
    try:
        import lpips as _lpips_lib
        loss_fn = _lpips_lib.LPIPS(net=net).to(pred.device)
        # lpips expects 3-channel RGB; use first 3 bands as proxy
        return loss_fn(pred[:, :3], target[:, :3]).mean()
    except ImportError:
        # Graceful degradation
        return mae(pred[:, :3], target[:, :3])


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    cloud_mask: Optional[torch.Tensor] = None,
    use_lpips: bool = False,
) -> Dict[str, float]:
    """Compute all metrics for a batch.

    If *cloud_mask* is provided, also reports cloud-region–specific MAE.

    Args:
        pred, target:  (B, C, H, W) in [-1, 1].
        cloud_mask:    (B, 1, H, W) binary mask (1 = cloud), or None.
        use_lpips:     Whether to compute LPIPS (slow on CPU).

    Returns:
        Dict mapping metric names to Python floats.
    """
    metrics = {
        "psnr":  psnr(pred, target).item(),
        "ssim":  ssim(pred, target).item(),
        "mae":   mae(pred, target).item(),
        "sam":   sam(pred, target).item(),
    }

    if cloud_mask is not None:
        cloud_pred = pred * cloud_mask
        cloud_tgt = target * cloud_mask
        n_cloud = cloud_mask.sum().clamp(min=1.0)
        metrics["cloud_mae"] = (cloud_pred - cloud_tgt).abs().sum().item() / n_cloud.item()

    if use_lpips:
        metrics["lpips"] = lpips(pred, target).item()

    return metrics


# ---------------------------------------------------------------------------
# Running aggregator
# ---------------------------------------------------------------------------

class MetricAggregator:
    """Accumulates per-batch metrics and computes final averages."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._count: int = 0

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + v
        self._count += 1

    def compute(self) -> Dict[str, float]:
        if self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sums.items()}

    def reset(self) -> None:
        self._sums.clear()
        self._count = 0
