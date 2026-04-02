"""Visualisation utilities: before/after comparison plots and band inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")   # headless backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_rgb(
    tensor: torch.Tensor,
    rgb_bands: Tuple[int, int, int] = (3, 2, 1),
    percentile: float = 2.0,
) -> np.ndarray:
    """Convert a (C, H, W) tensor to a (H, W, 3) uint8 RGB array.

    Args:
        tensor:     Single-image tensor in [-1, 1].
        rgb_bands:  Band indices to use as R, G, B (default: Sentinel-2 B4, B3, B2).
        percentile: Clip low/high percentile for contrast stretching.
    """
    img = tensor.detach().cpu().float()
    rgb = torch.stack([img[b] for b in rgb_bands], dim=0)   # (3, H, W)

    # De-normalise from [-1, 1] → [0, 1]
    rgb = (rgb + 1.0) / 2.0
    arr = rgb.permute(1, 2, 0).numpy()                       # (H, W, 3)

    # Percentile contrast stretching
    lo = np.percentile(arr, percentile)
    hi = np.percentile(arr, 100.0 - percentile)
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    return (arr * 255).astype(np.uint8)


def _to_numpy_gray(tensor: torch.Tensor, band: int = 0) -> np.ndarray:
    """Convert a single band of a (C, H, W) tensor to a (H, W) uint8 array."""
    arr = tensor[band].detach().cpu().float().numpy()
    arr = (arr + 1.0) / 2.0
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_comparison_grid(
    cloudy: torch.Tensor,
    reconstructed: torch.Tensor,
    clear: torch.Tensor,
    save_path: str | Path,
    cloud_mask: Optional[torch.Tensor] = None,
    rgb_bands: Tuple[int, int, int] = (3, 2, 1),
    title: str = "",
) -> None:
    """Save a side-by-side comparison: cloudy | reconstructed | clear [| mask].

    Args:
        cloudy:         (C, H, W) cloudy input image.
        reconstructed:  (C, H, W) network output.
        clear:          (C, H, W) ground-truth clear image.
        save_path:      Output PNG file path.
        cloud_mask:     Optional (1, H, W) binary cloud mask.
        rgb_bands:      Band indices for RGB visualisation.
        title:          Optional figure title.
    """
    if not _MPL:
        raise ImportError("matplotlib is required for visualisation: pip install matplotlib")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_cols = 4 if cloud_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(_to_numpy_rgb(cloudy, rgb_bands))
    axes[0].set_title("Cloudy Input")
    axes[0].axis("off")

    axes[1].imshow(_to_numpy_rgb(reconstructed, rgb_bands))
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    axes[2].imshow(_to_numpy_rgb(clear, rgb_bands))
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    if cloud_mask is not None:
        mask_arr = cloud_mask[0].detach().cpu().numpy()
        axes[3].imshow(mask_arr, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Cloud Mask")
        axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_band(
    tensor: torch.Tensor,
    band_idx: int,
    save_path: str | Path,
    cmap: str = "gray",
    title: str = "",
) -> None:
    """Save a single spectral band as a grayscale image.

    Args:
        tensor:    (C, H, W) image tensor.
        band_idx:  Which band to render.
        save_path: Output path.
        cmap:      Matplotlib colormap.
        title:     Optional title.
    """
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _to_numpy_gray(tensor, band=band_idx)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(
    metric_history: dict,
    save_path: str | Path,
    title: str = "Training Metrics",
) -> None:
    """Plot one curve per key in *metric_history* over training steps.

    Args:
        metric_history: Dict mapping metric name → list of float values.
        save_path:      Output PNG path.
        title:          Figure title.
    """
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, values in metric_history.items():
        ax.plot(values, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
