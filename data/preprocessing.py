"""Band clipping, normalization, and patch extraction utilities."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Per-sensor clip ranges (empirical percentiles on SEN12MS-CR)
# ---------------------------------------------------------------------------

S2_CLIP_MIN = 0.0
S2_CLIP_MAX = 10000.0   # surface reflectance scaled x10000

S1_CLIP_MIN = -25.0     # dB
S1_CLIP_MAX = 0.0       # dB


# ---------------------------------------------------------------------------
# Normalisation / de-normalisation
# ---------------------------------------------------------------------------

def clip_and_normalize(
    tensor: torch.Tensor,
    clip_min: float,
    clip_max: float,
    target_min: float = -1.0,
    target_max: float = 1.0,
) -> torch.Tensor:
    """Clip values to [clip_min, clip_max] then linearly scale to [target_min, target_max]."""
    tensor = tensor.clamp(clip_min, clip_max)
    tensor = (tensor - clip_min) / (clip_max - clip_min)          # [0, 1]
    tensor = tensor * (target_max - target_min) + target_min      # [target_min, target_max]
    return tensor


def normalize_s2(x: torch.Tensor) -> torch.Tensor:
    """Normalize Sentinel-2 reflectance to [-1, 1]."""
    return clip_and_normalize(x, S2_CLIP_MIN, S2_CLIP_MAX)


def normalize_s1(x: torch.Tensor) -> torch.Tensor:
    """Normalize Sentinel-1 dB values to [-1, 1]."""
    return clip_and_normalize(x, S1_CLIP_MIN, S1_CLIP_MAX)


def denormalize(
    tensor: torch.Tensor,
    clip_min: float,
    clip_max: float,
    source_min: float = -1.0,
    source_max: float = 1.0,
) -> torch.Tensor:
    """Invert clip_and_normalize — map [source_min, source_max] → [clip_min, clip_max]."""
    tensor = (tensor - source_min) / (source_max - source_min)    # [0, 1]
    tensor = tensor * (clip_max - clip_min) + clip_min
    return tensor


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(
    image: torch.Tensor,
    patch_size: int,
    overlap: int = 0,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Slice a (C, H, W) image into overlapping patches.

    Returns:
        patches: (N, C, patch_size, patch_size) tensor.
        grid:    (n_rows, n_cols) tuple for reconstruction.
    """
    _, H, W = image.shape
    stride = patch_size - overlap

    patches = []
    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            patch = image[:, top : top + patch_size, left : left + patch_size]
            patches.append(patch)

    n_rows = len(range(0, H - patch_size + 1, stride))
    n_cols = len(range(0, W - patch_size + 1, stride))

    return torch.stack(patches, dim=0), (n_rows, n_cols)


def reconstruct_from_patches(
    patches: torch.Tensor,
    original_size: Tuple[int, int],
    patch_size: int,
    overlap: int = 0,
) -> torch.Tensor:
    """Reconstruct image from overlapping patches using linear blending.

    Args:
        patches:       (N, C, patch_size, patch_size)
        original_size: (H, W) of the target image.
        patch_size:    Side length of each patch.
        overlap:       Overlap in pixels between adjacent patches.

    Returns:
        Reconstructed (C, H, W) tensor.
    """
    H, W = original_size
    stride = patch_size - overlap
    C = patches.shape[1]

    output = torch.zeros(C, H, W, dtype=patches.dtype)
    weight = torch.zeros(1, H, W, dtype=patches.dtype)

    # Build a Hann window for smooth blending
    window_1d = torch.hann_window(patch_size, periodic=False)
    window_2d = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)   # (patch_size, patch_size)
    window_2d = window_2d.unsqueeze(0)                            # (1, patch_size, patch_size)

    idx = 0
    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            output[:, top : top + patch_size, left : left + patch_size] += (
                patches[idx] * window_2d
            )
            weight[:, top : top + patch_size, left : left + patch_size] += window_2d
            idx += 1

    weight = weight.clamp(min=1e-8)
    return output / weight


# ---------------------------------------------------------------------------
# Batch-level preprocessing transform (for DataLoader use)
# ---------------------------------------------------------------------------

class PreprocessTransform:
    """Callable transform: normalizes s1 + s2 bands in a sample dict."""

    def __call__(self, sample: dict) -> dict:
        if "s1" in sample:
            sample["s1"] = normalize_s1(sample["s1"])
        if "s2_cloudy" in sample:
            sample["s2_cloudy"] = normalize_s2(sample["s2_cloudy"])
        if "s2_clear" in sample:
            sample["s2_clear"] = normalize_s2(sample["s2_clear"])
        return sample
