"""Preprocessing utilities for SEN12MS-CR data.

All functions handle both single tensors and batched tensors:
  - Single:  (C, H, W)
  - Batched: (B, C, H, W)

Sentinel-2 band order (13-band L1C stack, 0-indexed):
  0:B1  1:B2  2:B3  3:B4  4:B5  5:B6  6:B7
  7:B8  8:B8A 9:B9  9:B10 10:B11 11:B12
"""

from __future__ import annotations

from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel-2 raw reflectance range
_S2_RAW_MIN: float = 0.0
_S2_RAW_MAX: float = 10000.0

# Sentinel-1 per-band dB clip ranges
# index 0 = VV, index 1 = VH
_SAR_CLIP_RANGES: dict[int, tuple[float, float]] = {
    0: (-25.0,  0.0),   # VV
    1: (-32.5,  0.0),   # VH
}
_SAR_CLIP_DEFAULT: tuple[float, float] = (-25.0, 0.0)

# Sentinel-2 RGB band indices (B4=red, B3=green, B2=blue)
_S2_RGB_BANDS: tuple[int, int, int] = (3, 2, 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_batched(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 4


def _ensure_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float() if tensor.dtype != torch.float32 else tensor


# ---------------------------------------------------------------------------
# 1. normalize_optical
# ---------------------------------------------------------------------------

def normalize_optical(
    tensor: torch.Tensor,
    method: Literal["minmax", "percentile"] = "minmax",
    percentile: float = 2.0,
) -> torch.Tensor:
    """Normalise Sentinel-2 optical data from raw reflectance to [0, 1].

    Args:
        tensor:     (C, H, W) or (B, C, H, W) float tensor with raw S2
                    reflectance values in [0, 10000].
        method:     ``"minmax"``   — clip to [0, 10000] then divide by 10000.
                    ``"percentile"``— per-image percentile stretch; robust to
                    saturated pixels, useful for visualisation.
        percentile: Low/high percentile used when ``method="percentile"``.
                    E.g. 2.0 clips the bottom and top 2 % of values.

    Returns:
        Tensor of same shape in [0, 1], float32.

    Examples:
        >>> img = torch.randint(0, 10001, (13, 256, 256)).float()
        >>> out = normalize_optical(img)
        >>> out.min() >= 0 and out.max() <= 1
        True
    """
    tensor = _ensure_float(tensor)

    if method == "minmax":
        return tensor.clamp(_S2_RAW_MIN, _S2_RAW_MAX) / _S2_RAW_MAX

    if method == "percentile":
        # Compute percentiles over spatial dims; keep C (and B) dims intact
        flat = tensor.flatten(-2)  # (..., C, H*W)
        lo = torch.quantile(flat, percentile / 100.0, dim=-1, keepdim=True)
        hi = torch.quantile(flat, 1.0 - percentile / 100.0, dim=-1, keepdim=True)
        lo = lo.unsqueeze(-1)   # (..., C, 1, 1)
        hi = hi.unsqueeze(-1)
        denom = (hi - lo).clamp(min=1e-6)
        return ((tensor - lo) / denom).clamp(0.0, 1.0)

    raise ValueError(f"Unknown method '{method}'. Choose 'minmax' or 'percentile'.")


# ---------------------------------------------------------------------------
# 2. normalize_sar
# ---------------------------------------------------------------------------

def normalize_sar(tensor: torch.Tensor) -> torch.Tensor:
    """Normalise Sentinel-1 SAR data to [0, 1].

    Per-band processing:
      1. Clip to the empirical dB range (VV: [-25, 0], VH: [-32.5, 0]).
      2. Shift to positive domain: subtract the clip minimum.
      3. Scale by the band range so output is in [0, 1].

    Any bands beyond index 1 fall back to the VV clip range.

    Args:
        tensor: (C, H, W) or (B, C, H, W) float tensor with SAR backscatter
                in dB (typically negative values).

    Returns:
        Tensor of same shape in [0, 1], float32.

    Examples:
        >>> sar = torch.tensor([-20.0, -30.0]).reshape(2, 1, 1)
        >>> out = normalize_sar(sar)
        >>> out.shape
        torch.Size([2, 1, 1])
    """
    tensor = _ensure_float(tensor)
    batched = _is_batched(tensor)
    t = tensor if batched else tensor.unsqueeze(0)   # (B, C, H, W)

    bands = []
    for i in range(t.shape[1]):
        lo, hi = _SAR_CLIP_RANGES.get(i, _SAR_CLIP_DEFAULT)
        band = t[:, i : i + 1].clamp(lo, hi)
        band = (band - lo) / (hi - lo)              # → [0, 1]
        bands.append(band)

    out = torch.cat(bands, dim=1)
    return out if batched else out.squeeze(0)


# ---------------------------------------------------------------------------
# 3. to_diffusion_range
# ---------------------------------------------------------------------------

def to_diffusion_range(tensor: torch.Tensor) -> torch.Tensor:
    """Map values from [0, 1] → [-1, 1] for diffusion model training.

    The VQ-GAN decoder uses Tanh output and the diffusion bridge operates in
    [-1, 1] latent space.  Apply this after ``normalize_optical`` /
    ``normalize_sar`` before passing tensors to the model.

    Args:
        tensor: Any shape float tensor with values in [0, 1].

    Returns:
        Tensor of same shape in [-1, 1].

    Examples:
        >>> x = torch.tensor([0.0, 0.5, 1.0])
        >>> to_diffusion_range(x)
        tensor([-1.,  0.,  1.])
    """
    return tensor.float() * 2.0 - 1.0


# ---------------------------------------------------------------------------
# 4. from_diffusion_range
# ---------------------------------------------------------------------------

def from_diffusion_range(tensor: torch.Tensor) -> torch.Tensor:
    """Map values from [-1, 1] → [0, 1] for evaluation / visualisation.

    Inverse of :func:`to_diffusion_range`.  Apply to model outputs before
    computing metrics or saving images.

    Args:
        tensor: Any shape float tensor with values in [-1, 1].

    Returns:
        Tensor of same shape in [0, 1], clamped.

    Examples:
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> from_diffusion_range(x)
        tensor([0., 0.5000, 1.])
    """
    return ((tensor.float() + 1.0) / 2.0).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# 5. extract_rgb
# ---------------------------------------------------------------------------

def extract_rgb(
    tensor: torch.Tensor,
    bands: Tuple[int, int, int] = _S2_RGB_BANDS,
    to_uint8: bool = False,
) -> torch.Tensor:
    """Extract RGB channels from a 13-band Sentinel-2 tensor.

    Defaults to (B4, B3, B2) = (red, green, blue) in standard RGB order,
    which matches how most visualisation tools expect the channel order.

    Args:
        tensor:   (C, H, W) or (B, C, H, W) Sentinel-2 tensor.
                  Values should already be in [0, 1] — call
                  ``normalize_optical`` first if working with raw data.
        bands:    Three band indices to extract as (R, G, B).
                  Default: ``(3, 2, 1)`` = B4, B3, B2.
        to_uint8: If True, scale [0, 1] → [0, 255] and cast to uint8.
                  Useful for saving PNG/JPEG images.

    Returns:
        (3, H, W) or (B, 3, H, W) RGB tensor, float32 in [0, 1]
        (or uint8 in [0, 255] when ``to_uint8=True``).

    Examples:
        >>> img = torch.rand(13, 256, 256)
        >>> rgb = extract_rgb(img)
        >>> rgb.shape
        torch.Size([3, 256, 256])
    """
    tensor = _ensure_float(tensor)
    batched = _is_batched(tensor)
    t = tensor if batched else tensor.unsqueeze(0)   # (B, C, H, W)

    r, g, b = bands
    rgb = torch.stack([t[:, r], t[:, g], t[:, b]], dim=1)  # (B, 3, H, W)
    rgb = rgb.clamp(0.0, 1.0)

    if to_uint8:
        rgb = (rgb * 255.0).to(torch.uint8)

    return rgb if batched else rgb.squeeze(0)


# ---------------------------------------------------------------------------
# 6. patchify
# ---------------------------------------------------------------------------

def patchify(
    tensor: torch.Tensor,
    patch_size: int = 256,
    overlap: int = 32,
) -> Tuple[torch.Tensor, dict]:
    """Split a large image into overlapping patches.

    Works on single images (C, H, W) and batches (B, C, H, W).
    For batches, all images are patchified independently and stacked.

    Args:
        tensor:     (C, H, W) or (B, C, H, W) float tensor.
        patch_size: Side length of each square patch in pixels.
        overlap:    Overlap between adjacent patches in pixels.
                    Must be < patch_size.

    Returns:
        patches:  (N, C, P, P) for single or (B*N, C, P, P) for batched,
                  where P = patch_size.
        meta:     Dict with reconstruction metadata:
                    - ``original_shape``: (H, W) or (B, H, W)
                    - ``patch_size``:     int
                    - ``overlap``:        int
                    - ``grid``:           (n_rows, n_cols)
                    - ``batched``:        bool

    Raises:
        ValueError: If overlap >= patch_size.

    Examples:
        >>> img = torch.rand(13, 512, 512)
        >>> patches, meta = patchify(img, patch_size=256, overlap=32)
        >>> patches.shape   # 3×3 grid of 256×256 patches
        torch.Size([9, 13, 256, 256])
    """
    if overlap >= patch_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than patch_size ({patch_size})."
        )

    batched = _is_batched(tensor)
    t = tensor if batched else tensor.unsqueeze(0)   # (B, C, H, W)
    B, C, H, W = t.shape
    stride = patch_size - overlap

    # Pad so dimensions are covered completely
    pad_h = (stride - (H - patch_size) % stride) % stride
    pad_w = (stride - (W - patch_size) % stride) % stride
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_pad, W_pad = t.shape

    row_starts = list(range(0, H_pad - patch_size + 1, stride))
    col_starts = list(range(0, W_pad - patch_size + 1, stride))
    n_rows, n_cols = len(row_starts), len(col_starts)

    all_patches = []
    for top in row_starts:
        for left in col_starts:
            p = t[:, :, top : top + patch_size, left : left + patch_size]
            all_patches.append(p)           # each (B, C, P, P)

    # Stack: (n_patches, B, C, P, P) → (B*n_patches, C, P, P)
    patches = torch.stack(all_patches, dim=1)           # (B, N, C, P, P)
    patches = patches.reshape(B * n_rows * n_cols, C, patch_size, patch_size)

    meta = {
        "original_shape": (B, H, W) if batched else (H, W),
        "padded_shape":   (H_pad, W_pad),
        "patch_size":     patch_size,
        "overlap":        overlap,
        "grid":           (n_rows, n_cols),
        "batch_size":     B,
        "batched":        batched,
    }
    return patches, meta


# ---------------------------------------------------------------------------
# 7. unpatchify
# ---------------------------------------------------------------------------

def _linear_blend_weights(patch_size: int, overlap: int) -> torch.Tensor:
    """Build a (1, 1, P, P) linear blend weight window.

    Within the interior (non-overlap) region the weight is 1.0.
    In the overlap margins the weight ramps linearly from 0 → 1,
    so blended seams are smooth.
    """
    stride = patch_size - overlap
    w = torch.ones(patch_size)

    if overlap > 0:
        ramp = torch.linspace(0.0, 1.0, overlap)
        w[:overlap]  = ramp          # left / top fade-in
        w[-overlap:] = ramp.flip(0)  # right / bottom fade-out

    window_2d = w.unsqueeze(0) * w.unsqueeze(1)   # (P, P)
    return window_2d.reshape(1, 1, patch_size, patch_size)


def unpatchify(
    patches: torch.Tensor,
    meta: dict,
) -> torch.Tensor:
    """Reassemble overlapping patches into the original image using linear blending.

    Linear blending in overlap regions avoids hard seams: each patch is
    multiplied by a weight window that ramps 0→1 at its edges before
    accumulation, then divided by the summed weights.

    Args:
        patches: (N, C, P, P) or (B*N, C, P, P) tensor of patches as
                 returned by :func:`patchify`.
        meta:    Metadata dict returned by :func:`patchify`.

    Returns:
        Reconstructed tensor:
          - (C, H, W) if original input was single.
          - (B, C, H, W) if original input was batched.

    Examples:
        >>> img = torch.rand(13, 512, 512)
        >>> patches, meta = patchify(img, patch_size=256, overlap=32)
        >>> rec = unpatchify(patches, meta)
        >>> rec.shape
        torch.Size([13, 512, 512])
    """
    patch_size  = meta["patch_size"]
    overlap     = meta["overlap"]
    n_rows, n_cols = meta["grid"]
    H_pad, W_pad   = meta["padded_shape"]
    B               = meta["batch_size"]
    batched         = meta["batched"]
    orig_shape      = meta["original_shape"]

    C = patches.shape[1]
    N = n_rows * n_cols

    # Reshape back to (B, N, C, P, P)
    patches = patches.reshape(B, N, C, patch_size, patch_size)

    canvas  = torch.zeros(B, C, H_pad, W_pad, dtype=patches.dtype, device=patches.device)
    weights = torch.zeros(B, 1, H_pad, W_pad, dtype=patches.dtype, device=patches.device)

    window = _linear_blend_weights(patch_size, overlap).to(patches.device)  # (1,1,P,P)
    stride = patch_size - overlap

    idx = 0
    row_starts = list(range(0, H_pad - patch_size + 1, stride))
    col_starts = list(range(0, W_pad - patch_size + 1, stride))

    for top in row_starts:
        for left in col_starts:
            p = patches[:, idx]                           # (B, C, P, P)
            canvas [:, :, top : top + patch_size, left : left + patch_size] += p * window
            weights[:, :, top : top + patch_size, left : left + patch_size] += window
            idx += 1

    canvas = canvas / weights.clamp(min=1e-8)

    # Crop back to original (un-padded) size
    if batched:
        _, H, W = orig_shape
    else:
        H, W = orig_shape
    canvas = canvas[:, :, :H, :W]

    return canvas if batched else canvas.squeeze(0)


# ---------------------------------------------------------------------------
# Convenience: combined pipeline transforms used by DataLoader
# ---------------------------------------------------------------------------

class PreprocessTransform:
    """Callable transform applied to a sample dict from SEN12MSCRDataset.

    Assumes tensors are already in [0, 1] (normalised in the dataset loader).
    Optionally maps to [-1, 1] for diffusion training.

    Args:
        diffusion_range: If True, apply :func:`to_diffusion_range` to optical
                         tensors (SAR is kept in [0, 1] as conditioning input).
    """

    def __init__(self, diffusion_range: bool = False) -> None:
        self.diffusion_range = diffusion_range

    def __call__(self, sample: dict) -> dict:
        for key in ("cloudy", "clean"):
            if key in sample:
                t = sample[key]
                if self.diffusion_range:
                    t = to_diffusion_range(t)
                sample[key] = t
        # SAR stays in [0, 1] — it's used as a conditioning signal, not diffused
        return sample


# ---------------------------------------------------------------------------
# Keep old names as thin aliases for backwards compatibility
# ---------------------------------------------------------------------------

def clip_and_normalize(
    tensor: torch.Tensor,
    clip_min: float,
    clip_max: float,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> torch.Tensor:
    """Generic clip-and-scale utility (backwards-compatible alias)."""
    tensor = tensor.float().clamp(clip_min, clip_max)
    tensor = (tensor - clip_min) / (clip_max - clip_min)
    if target_min != 0.0 or target_max != 1.0:
        tensor = tensor * (target_max - target_min) + target_min
    return tensor


def normalize_s2(x: torch.Tensor) -> torch.Tensor:
    """Alias: normalize Sentinel-2 reflectance to [0, 1]."""
    return normalize_optical(x, method="minmax")


def normalize_s1(x: torch.Tensor) -> torch.Tensor:
    """Alias: normalize Sentinel-1 dB values to [0, 1]."""
    return normalize_sar(x)


def extract_patches(
    image: torch.Tensor,
    patch_size: int,
    overlap: int = 0,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Backwards-compatible alias for :func:`patchify`.

    Returns (patches, (n_rows, n_cols)) to match the old interface.
    """
    patches, meta = patchify(image, patch_size=patch_size, overlap=overlap)
    return patches, meta["grid"]


def reconstruct_from_patches(
    patches: torch.Tensor,
    original_size: Tuple[int, int],
    patch_size: int,
    overlap: int = 0,
) -> torch.Tensor:
    """Backwards-compatible alias for :func:`unpatchify`."""
    H, W = original_size
    n_rows = (H - patch_size) // max(patch_size - overlap, 1) + 1
    n_cols = (W - patch_size) // max(patch_size - overlap, 1) + 1
    meta = {
        "original_shape": (H, W),
        "padded_shape":   (H, W),
        "patch_size":     patch_size,
        "overlap":        overlap,
        "grid":           (n_rows, n_cols),
        "batch_size":     1,
        "batched":        False,
    }
    return unpatchify(patches, meta)
