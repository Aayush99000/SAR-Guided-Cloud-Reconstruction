"""Cloud mask generation utilities — Sen2Cor-inspired approach.

Band index mapping for SEN12MS-CR (0-indexed, 13-band L1C stack):
    0:B1  (coastal aerosol, 443 nm)
    1:B2  (blue,  490 nm)
    2:B3  (green, 560 nm)
    3:B4  (red,   665 nm)
    4:B5  (red-edge 1, 705 nm)
    5:B6  (red-edge 2, 740 nm)
    6:B7  (red-edge 3, 783 nm)
    7:B8  (NIR broad,  842 nm)
    8:B8A (NIR narrow, 865 nm)
    9:B9  (water vapour, 945 nm)
    9:B10 (SWIR cirrus, 1375 nm)  ← index 9 in the 13-band stack
   10:B11 (SWIR-1, 1610 nm)       ← index 10
   11:B12 (SWIR-2, 2190 nm)       ← index 11

Note: Some SEN12MS-CR distributions merge B9 and B10 into position 9,
making B11→10, B12→11. The constants below follow this convention.
All input tensors are expected in [0, 1] (pre-scaled from raw reflectance).
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band index constants
# ---------------------------------------------------------------------------
_B1  = 0    # coastal aerosol
_B2  = 1    # blue
_B3  = 2    # green
_B4  = 3    # red
_B10 = 9    # SWIR cirrus
_B11 = 10   # SWIR-1

# Sen2Cor cloud probability thresholds
_T_CLOUD: float = 0.2
_T_NDSI:  float = 0.6      # NDSI above this → snow, not cloud

# Minimum number of bands required for the full algorithm
_MIN_BANDS_FULL: int = 11  # need up to index 10 (B11)
_MIN_BANDS_NDSI: int = 11


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_divide(
    num: torch.Tensor,
    denom: torch.Tensor,
    fill: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Element-wise division that replaces near-zero denominators with *fill*."""
    mask = denom.abs() < eps
    result = torch.where(mask, torch.full_like(num, fill), num / (denom + eps * mask.float()))
    return result


def _linear_scale(
    x: torch.Tensor,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """Map x from [lo, hi] → [0, 1], clamped to [-inf, 1].

    Values below *lo* become negative (treated as non-cloud signal).
    Values above *hi* saturate at 1.
    No lower clamp is applied so that sub-threshold bands do not artificially
    boost Sc via the min() aggregation.
    """
    denom = hi - lo
    if abs(denom) < 1e-8:
        return torch.zeros_like(x)
    return (x - lo) / denom


def _ensure_float32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float32 else t.float()


def _validate_input(s2: torch.Tensor) -> None:
    if s2.ndim != 3:
        raise ValueError(
            f"Expected (C, H, W) tensor, got shape {tuple(s2.shape)}"
        )
    if s2.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"Expected float tensor, got {s2.dtype}")


# ---------------------------------------------------------------------------
# Cloud probability score (Sen2Cor-inspired)
# ---------------------------------------------------------------------------

def _cloud_probability_score(s2: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel cloud probability Sc ∈ (-inf, 1].

    Uses four spectral tests combined via element-wise min:

      T1  = (B2  - 0.1) / (0.5 - 0.1)        — blue band brightness
      T2  = (B1  - 0.1) / (0.3 - 0.1)        — coastal aerosol
      T3  = (B10 + B1 - 0.15) / (0.2 - 0.15) — cirrus + aerosol combo
      T4  = (B4  + B3  + B2  - 0.2) / (0.8 - 0.2)  — visible band sum

      Sc = min(T1, T2, T3, T4)

    Args:
        s2: (C, H, W) in [0, 1].  Requires C >= 11.

    Returns:
        (H, W) float32 score tensor.
    """
    B1  = s2[_B1]
    B2  = s2[_B2]
    B3  = s2[_B3]
    B4  = s2[_B4]
    B10 = s2[_B10]

    T1 = _linear_scale(B2,               lo=0.10, hi=0.50)   # blue brightness
    T2 = _linear_scale(B1,               lo=0.10, hi=0.30)   # coastal aerosol
    T3 = _linear_scale(B10 + B1,         lo=0.15, hi=0.20)   # cirrus + aerosol
    T4 = _linear_scale(B4 + B3 + B2,     lo=0.20, hi=0.80)   # visible sum

    Sc = torch.stack([T1, T2, T3, T4], dim=0).min(dim=0).values   # (H, W)

    # Sanitise any NaN/Inf that slipped through (e.g. saturated sensor pixels)
    Sc = torch.nan_to_num(Sc, nan=0.0, posinf=1.0, neginf=0.0)
    return Sc


# ---------------------------------------------------------------------------
# NDSI — snow discrimination
# ---------------------------------------------------------------------------

def _ndsi(s2: torch.Tensor) -> torch.Tensor:
    """Normalised Difference Snow Index: (B3 - B11) / (B3 + B11).

    Returns:
        (H, W) float32, clamped to [-1, 1].
    """
    B3  = s2[_B3]
    B11 = s2[_B11]
    ndsi = _safe_divide(B3 - B11, B3 + B11, fill=0.0)
    return ndsi.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_cloud_mask(
    s2: torch.Tensor,
    t_cloud: float = _T_CLOUD,
    t_ndsi:  float = _T_NDSI,
    dilate:  int   = 0,
    brightness_threshold: float = 0.35,   # fallback only
) -> torch.Tensor:
    """Generate a binary cloud mask using a Sen2Cor-inspired spectral test.

    **Algorithm:**

    1. Compute multi-band cloud probability score Sc via four linear spectral
       tests (blue brightness, coastal aerosol, cirrus + aerosol, visible sum).
    2. Threshold at *t_cloud* → binary mask M.
    3. Compute NDSI and suppress snow pixels (NDSI > *t_ndsi*) → M'.
    4. Optionally dilate M' by *dilate* pixels.

    Falls back to a simple visible-band brightness threshold when fewer than
    11 bands are available (e.g. RGB+NIR-only stacks).

    Args:
        s2:                   (C, H, W) float32 Sentinel-2 tensor in [0, 1].
        t_cloud:              Cloud probability threshold (default 0.2).
        t_ndsi:               NDSI threshold above which pixels are classified
                              as snow and removed from the cloud mask (default 0.6).
        dilate:               Morphological dilation radius in pixels (0 = off).
        brightness_threshold: Threshold used by the simple fallback method.

    Returns:
        (1, H, W) float32 binary cloud mask — 1 = cloud, 0 = clear.
    """
    _validate_input(s2)
    s2 = _ensure_float32(s2)

    C = s2.shape[0]

    # ------------------------------------------------------------------
    # Full Sen2Cor path (requires B1, B2, B3, B4, B10, B11)
    # ------------------------------------------------------------------
    if C >= _MIN_BANDS_FULL:
        Sc = _cloud_probability_score(s2)          # (H, W)
        M  = (Sc >= t_cloud).float()              # step 2

        # Step 3: remove snow
        ndsi_map = _ndsi(s2)
        snow     = (ndsi_map > t_ndsi).float()
        M_prime  = M * (1.0 - snow)              # clear snow pixels from mask

    # ------------------------------------------------------------------
    # Fallback: brightness heuristic (< 11 bands)
    # ------------------------------------------------------------------
    else:
        log.debug(
            "generate_cloud_mask: only %d bands available, using brightness fallback.", C
        )
        n = C
        # Use first three bands as visible-range proxy
        vis_idx = [min(1, n - 1), min(2, n - 1), min(3, n - 1)]
        visible = s2[vis_idx]

        # Normalise to [0, 1] regardless of accidental range
        if visible.min() < 0:
            visible = (visible + 1.0) / 2.0
        elif visible.max() > 1.0:
            visible = visible / 10000.0

        brightness = visible.mean(dim=0)
        M_prime = (brightness > brightness_threshold).float()

    # ------------------------------------------------------------------
    # Dilation
    # ------------------------------------------------------------------
    M_prime = M_prime.unsqueeze(0)                # (1, H, W)

    if dilate > 0:
        M_prime = dilate_cloud_mask(M_prime, dilation_pixels=dilate)

    return M_prime.clamp(0.0, 1.0)


def cloud_thickness_weight(
    cloudy: torch.Tensor,
    clean:  torch.Tensor,
    cloud_mask: torch.Tensor,
    I_cloud:  float = 1.0,
    I_ground: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-pixel cloud optical thickness weight α_c.

    Estimates how opaque each cloud pixel is by comparing the cloudy and
    clean observations against the theoretical cloud/ground radiance limits:

        α_c(i,j) = |cloudy(i,j) - clean(i,j)| / (I_cloud - I_ground)

    Result is averaged across spectral bands and clamped to [0, 1].
    Outside cloud pixels the weight is 0.

    Intended use: pass as a per-pixel multiplier inside the cloud-aware loss
    so that thick/opaque clouds receive higher gradient weight than thin cirrus.

    Args:
        cloudy:     (C, H, W) float32 cloudy image in [0, 1].
        clean:      (C, H, W) float32 cloud-free reference in [0, 1].
        cloud_mask: (1, H, W) float32 binary mask (1 = cloud).
        I_cloud:    Theoretical maximum radiance under full cloud (default 1.0).
        I_ground:   Theoretical ground radiance contribution (default 0.0).
        eps:        Small constant to avoid division by zero.

    Returns:
        (1, H, W) float32 weight map in [0, 1], zero outside cloud pixels.
    """
    if cloudy.ndim != 3 or clean.ndim != 3:
        raise ValueError("cloudy and clean must be (C, H, W) tensors.")
    if cloud_mask.ndim != 3 or cloud_mask.shape[0] != 1:
        raise ValueError("cloud_mask must be (1, H, W).")

    denom = abs(I_cloud - I_ground)
    if denom < eps:
        raise ValueError(
            f"I_cloud ({I_cloud}) and I_ground ({I_ground}) are too close — "
            "cannot compute meaningful thickness weight."
        )

    diff = (cloudy - clean).abs()              # (C, H, W)
    alpha = diff.mean(dim=0, keepdim=True)     # (1, H, W) — spectral average
    alpha = (alpha / (denom + eps)).clamp(0.0, 1.0)
    alpha = alpha * cloud_mask                 # zero outside cloud

    # Sanitise
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    return alpha


# ---------------------------------------------------------------------------
# Utility functions (kept from original interface)
# ---------------------------------------------------------------------------

def cloud_coverage_fraction(mask: torch.Tensor) -> float:
    """Fraction of pixels flagged as cloud in a (1, H, W) or (H, W) mask."""
    return float(mask.float().mean().item())


def dilate_cloud_mask(
    mask: torch.Tensor,
    dilation_pixels: int = 8,
) -> torch.Tensor:
    """Morphologically dilate a binary cloud mask via max-pooling.

    Args:
        mask:            (1, H, W) or (B, 1, H, W) float32 binary mask.
        dilation_pixels: Radius of the structuring element (pixels).

    Returns:
        Dilated mask with the same shape, values in {0, 1}.
    """
    squeeze = mask.ndim == 3
    if squeeze:
        mask = mask.unsqueeze(0)            # (1, 1, H, W)

    kernel_size = 2 * dilation_pixels + 1
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
    """Replace cloud pixels in *image* with *fill_value*.

    Args:
        image:      (C, H, W) image tensor.
        mask:       (1, H, W) binary mask (1 = cloud).
        fill_value: Value written to cloud pixels.

    Returns:
        (C, H, W) tensor with cloud pixels filled.
    """
    return image * (1.0 - mask) + fill_value * mask
