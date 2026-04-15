"""Image quality metrics for SAR-guided cloud removal evaluation.

All metric functions:
  - Accept (B, C, H, W) PyTorch tensors in [0, 1].
  - Return Python floats averaged over the batch.

Conventions
-----------
  ↑  higher is better   (PSNR, SSIM)
  ↓  lower is better    (MAE, SAM, LPIPS)

SAM is computed spectrally across all C channels; it is the primary metric
for multi-band fidelity in the remote-sensing community.

SSIM is implemented in pure PyTorch (equivalent to Wang et al. 2004 with
data_range=1.0).  If torchmetrics or skimage are installed they can be used
as drop-in alternatives; see the ``ssim`` docstring.

LPIPS requires the ``lpips`` package and operates on the first 3 channels
(RGB proxy).  A module-level cache avoids re-loading the VGG weights on
each call.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio averaged over the batch (dB).

    Args:
        pred, target: (B, C, H, W) in [0, max_val].
        max_val:      Dynamic range.  Default 1.0 for [0, 1] images.

    Returns:
        Mean PSNR across the batch in dB.  Returns ``inf`` when MSE = 0.
    """
    with torch.no_grad():
        mse = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))  # (B,)
        # Guard against log(0) while keeping identical-image case at inf
        psnr_per = torch.where(
            mse == 0,
            torch.full_like(mse, float("inf")),
            10.0 * torch.log10(max_val ** 2 / mse.clamp(min=1e-10)),
        )
        return psnr_per.mean().item()


# ---------------------------------------------------------------------------
# SSIM  (pure PyTorch, single-scale, Wang et al. 2004)
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """Structural Similarity Index averaged over batch and channels.

    Pure-PyTorch implementation equivalent to ``skimage.metrics.structural_similarity``
    with ``data_range=1.0, multichannel=True``.  To use torchmetrics instead::

        from torchmetrics.functional import structural_similarity_index_measure
        score = structural_similarity_index_measure(pred, target, data_range=1.0).item()

    Args:
        pred, target: (B, C, H, W) in [0, data_range].
        window_size:  Gaussian kernel size (default 11).
        data_range:   Dynamic range of the signal (default 1.0).

    Returns:
        Mean SSIM ∈ [0, 1].  1 = identical images.
    """
    with torch.no_grad():
        pred_f, target_f = pred.float(), target.float()
        C = pred_f.shape[1]
        kernel = _gaussian_kernel(window_size).to(pred_f.device, pred_f.dtype)
        kernel = kernel.expand(C, 1, window_size, window_size).contiguous()
        pad = window_size // 2

        mu1 = F.conv2d(pred_f,   kernel, padding=pad, groups=C)
        mu2 = F.conv2d(target_f, kernel, padding=pad, groups=C)

        mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = F.conv2d(pred_f   * pred_f,   kernel, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target_f * target_f, kernel, padding=pad, groups=C) - mu2_sq
        sigma12   = F.conv2d(pred_f   * target_f, kernel, padding=pad, groups=C) - mu12

        # Stability constants for data_range=1.0  (K1=0.01, K2=0.03 as in the paper)
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        num   = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
        denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
        return (num / denom).mean().item()


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error averaged over all elements in the batch."""
    with torch.no_grad():
        return (pred.float() - target.float()).abs().mean().item()


# ---------------------------------------------------------------------------
# SAM  (Spectral Angle Mapper)
# ---------------------------------------------------------------------------

def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Spectral Angle Mapper averaged over batch and spatial positions (degrees).

    SAM measures the angle between the predicted and reference spectral vectors
    at each pixel — independent of brightness, sensitive only to spectral shape.
    Lower is better; 0° = identical spectra.

    Formula:
        SAM(x, y) = arccos( dot(x, y) / (‖x‖ ‖y‖) )   [degrees]

    Args:
        pred, target: (B, C, H, W) — spectral dimension is C.
        eps:          Numerical floor for norms to avoid division by zero.

    Returns:
        Mean SAM across all pixels and batch samples in degrees.
    """
    with torch.no_grad():
        p, t = pred.float(), target.float()
        dot        = (p * t).sum(dim=1)                          # (B, H, W)
        norm_p     = p.norm(dim=1).clamp(min=eps)                # (B, H, W)
        norm_t     = t.norm(dim=1).clamp(min=eps)                # (B, H, W)
        cos_theta  = (dot / (norm_p * norm_t)).clamp(-1.0 + eps, 1.0 - eps)
        return torch.rad2deg(torch.acos(cos_theta)).mean().item()


# ---------------------------------------------------------------------------
# LPIPS  (Learned Perceptual Image Patch Similarity)
# ---------------------------------------------------------------------------

# Module-level cache: keyed by (net_name, device_str) to avoid reloading VGG.
_lpips_cache: Dict[str, object] = {}


def _get_lpips_fn(net: str, device: torch.device):
    """Lazily load and cache an lpips.LPIPS model."""
    key = f"{net}:{device}"
    if key not in _lpips_cache:
        try:
            import lpips as _lpips_lib  # noqa: PLC0415
            fn = _lpips_lib.LPIPS(net=net, verbose=False).to(device)
            fn.eval()
            _lpips_cache[key] = fn
        except ImportError:
            _lpips_cache[key] = None
    return _lpips_cache[key]


def lpips_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "vgg",
) -> float:
    """Learned Perceptual Image Patch Similarity (lower is better).

    Operates on the **first 3 channels** of the input tensors (treated as RGB).
    Requires the ``lpips`` package::

        pip install lpips

    Falls back to RGB MAE if the package is not installed.

    Args:
        pred, target: (B, C, H, W) in [0, 1].  Only first 3 channels used.
        net:          Backbone network: ``"vgg"`` (default) or ``"alex"``.

    Returns:
        Mean LPIPS score across the batch.  Lower = more perceptually similar.
    """
    with torch.no_grad():
        # LPIPS internally expects [-1, 1]; rescale from [0, 1]
        p3 = pred[:, :3].float() * 2.0 - 1.0
        t3 = target[:, :3].float() * 2.0 - 1.0

        fn = _get_lpips_fn(net, pred.device)
        if fn is not None:
            return fn(p3, t3).mean().item()
        else:
            # Graceful fallback: RGB L1 (rough perceptual proxy)
            return (p3 - t3).abs().mean().item()


# ---------------------------------------------------------------------------
# Unified per-batch interface
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    cloud_mask: Optional[torch.Tensor] = None,
    use_lpips: bool = False,
    lpips_net: str = "vgg",
) -> Dict[str, float]:
    """Compute all metrics for a batch; return a flat dict of Python floats.

    Args:
        pred, target:  (B, C, H, W) in [0, 1].
        cloud_mask:    Optional (B, 1, H, W) binary mask (1=cloud, 0=clear).
                       When provided, adds ``psnr_cloud`` and ``mae_cloud``
                       (metrics restricted to cloud-covered pixels).
        use_lpips:     Whether to run LPIPS (slow; requires lpips package).
        lpips_net:     ``"vgg"`` or ``"alex"`` (only used when use_lpips=True).

    Returns:
        Dict with keys: psnr, ssim, mae, sam[, psnr_cloud, mae_cloud, lpips].
    """
    out: Dict[str, float] = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "mae":  mae(pred, target),
        "sam":  sam(pred, target),
    }

    if cloud_mask is not None:
        # Mask to cloud-only pixels
        m = cloud_mask.float()
        cloud_pred   = pred.float()  * m
        cloud_target = target.float() * m
        n_cloud = m.sum().clamp(min=1.0)
        out["psnr_cloud"] = psnr(cloud_pred, cloud_target)
        out["mae_cloud"]  = ((cloud_pred - cloud_target).abs().sum() / n_cloud).item()

    if use_lpips:
        out["lpips"] = lpips_score(pred, target, net=lpips_net)

    return out


# ---------------------------------------------------------------------------
# Per-sample metrics  (one dict per image in the batch)
# ---------------------------------------------------------------------------

def per_sample_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    cloud_mask: Optional[torch.Tensor] = None,
    use_lpips: bool = False,
    lpips_net: str = "vgg",
) -> List[Dict[str, float]]:
    """Compute metrics individually for each image in the batch.

    Args:
        pred, target:  (B, C, H, W) in [0, 1].
        cloud_mask:    Optional (B, 1, H, W) binary cloud mask.
        use_lpips:     Whether to run LPIPS per sample.
        lpips_net:     ``"vgg"`` or ``"alex"``.

    Returns:
        List of length B; each element is the same dict as ``compute_metrics``.
    """
    results: List[Dict[str, float]] = []
    B = pred.shape[0]
    for i in range(B):
        p = pred[i : i + 1]      # (1, C, H, W) — keep batch dim for conv ops
        t = target[i : i + 1]
        m = cloud_mask[i : i + 1] if cloud_mask is not None else None
        results.append(compute_metrics(p, t, m, use_lpips, lpips_net))
    return results


# ---------------------------------------------------------------------------
# Running aggregator  (global mean; stratification lives in evaluate.py)
# ---------------------------------------------------------------------------

class MetricAggregator:
    """Accumulates per-batch metric dicts and computes global averages.

    Example::

        agg = MetricAggregator()
        for batch_metrics in run_inference(...):
            agg.update(batch_metrics)
        print(agg.compute())
    """

    def __init__(self) -> None:
        self._sums:   Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """Add one dict of scalar metrics (from one batch or one sample)."""
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0) + float(v)
            self._counts[k] = self._counts.get(k, 0)  + 1

    def compute(self) -> Dict[str, float]:
        """Return mean of each accumulated metric."""
        return {k: self._sums[k] / max(self._counts[k], 1) for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def __len__(self) -> int:
        return max(self._counts.values(), default=0)
