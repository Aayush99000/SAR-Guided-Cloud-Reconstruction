"""Cloud mask generation utilities.

Supports:
  - Threshold-based mask from SCL (Scene Classification Layer) band index.
  - Simple brightness/NDSI heuristic when SCL is unavailable.
  - Morphological dilation to expand mask boundaries.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# SCL class values that correspond to cloud/shadow (Sentinel-2 L2A)
_SCL_CLOUD_CLASSES = {3, 8, 9, 10, 11}   # shadow, med, high, cirrus, snow/ice


def generate_cloud_mask(
    s2: torch.Tensor,
    scl_band_idx: int | None = None,
    brightness_threshold: float = 0.35,
) -> torch.Tensor:
    """Generate a binary cloud mask (1 = cloud, 0 = clear).

    Strategy:
      1. If *scl_band_idx* is given, use the SCL band values directly.
      2. Otherwise, estimate clouds from the mean of visible bands (B2-B4, indices 1-3
         in a 13-band Sentinel-2 stack) exceeding *brightness_threshold*.

    Args:
        s2:                  (C, H, W) Sentinel-2 tensor (values in [-1, 1] after norm,
                             or raw [0, 10000] before norm).
        scl_band_idx:        Band index of the SCL layer within *s2*, or None.
        brightness_threshold: Normalised brightness cutoff when SCL not available.

    Returns:
        (1, H, W) float32 binary mask.
    """
    if scl_band_idx is not None:
        scl = s2[scl_band_idx]                          # (H, W)
        mask = torch.zeros_like(scl)
        for cls in _SCL_CLOUD_CLASSES:
            mask = mask + (scl == cls).float()
        return mask.unsqueeze(0).clamp(0.0, 1.0)       # (1, H, W)

    # --- Brightness heuristic ---
    # Prefer first 3 bands of whatever stack is passed (handles both the full
    # 13-band stack and a pre-selected RGB+NIR subset).
    n_bands = s2.shape[0]
    visible_idx = [min(1, n_bands - 1), min(2, n_bands - 1), min(3, n_bands - 1)]
    visible_bands = s2[visible_idx]                     # (3, H, W)

    # Normalise to [0, 1] regardless of input range
    vmin = visible_bands.min()
    if vmin < 0:
        # [-1, 1] → [0, 1]
        visible_bands = (visible_bands + 1.0) / 2.0
    elif visible_bands.max() > 1.0:
        # raw reflectance [0, 10000] → [0, 1]
        visible_bands = visible_bands / 10000.0

    brightness = visible_bands.mean(dim=0)              # (H, W)
    mask = (brightness > brightness_threshold).float().unsqueeze(0)
    return mask                                         # (1, H, W)


def cloud_coverage_fraction(mask: torch.Tensor) -> float:
    """Return fraction of pixels flagged as cloud in a (1, H, W) or (H, W) mask."""
    return float(mask.float().mean().item())


def dilate_cloud_mask(
    mask: torch.Tensor,
    dilation_pixels: int = 8,
) -> torch.Tensor:
    """Morphologically dilate a binary cloud mask to cover cloud edges.

    Args:
        mask:             (1, H, W) or (B, 1, H, W) float32 mask.
        dilation_pixels:  Radius of the structuring element in pixels.

    Returns:
        Dilated mask with same shape.
    """
    squeeze = mask.ndim == 3
    if squeeze:
        mask = mask.unsqueeze(0)                        # → (1, 1, H, W)

    kernel_size = 2 * dilation_pixels + 1
    # Max-pool approximates morphological dilation
    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=dilation_pixels,
    )

    if squeeze:
        dilated = dilated.squeeze(0)
    return dilated.clamp(0.0, 1.0)


def apply_cloud_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Zero-out (or fill) cloud pixels in *image* according to *mask*.

    Args:
        image:      (C, H, W) image tensor.
        mask:       (1, H, W) binary mask (1 = cloud).
        fill_value: Value to place at cloud pixels.

    Returns:
        Masked image (C, H, W).
    """
    return image * (1.0 - mask) + fill_value * mask
