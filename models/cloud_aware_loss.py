"""Adaptive cloud-weighted reconstruction loss.

The loss up-weights pixels in cloud-covered regions so the network focuses on
reconstructing the hardest (and most valuable) areas while still maintaining
structural coherence in clear regions.

Weight map formula:
    W(x,y) = alpha * M'(x,y) + (1 − alpha) * (1 − M'(x,y))

With alpha=0.8 (default):
    Cloud pixels  (M'=1) → W = 0.80   (~80% of gradient budget)
    Clear pixels  (M'=0) → W = 0.20   (enough to prevent background drift)

If cloud_thickness (continuous [0,1]) is supplied, the cloud term becomes
thickness-weighted so partially-cloudy pixels get proportionally less focus:
    W(x,y) = alpha * thickness(x,y) + (1 − alpha) * (1 − M'(x,y))

Components:
    L_final = mean( W * (lambda_mse * L_MSE + lambda_ssim * L_SSIM) )
    L_MSE   = |pred − target|²      (per-pixel)
    L_SSIM  = 1 − SSIM(pred, target) (per-pixel structural)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable per-pixel SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1-D Gaussian → outer product → (1,1,size,size) kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def _ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Per-pixel SSIM map (B, C, H, W).

    Values close to 1 indicate high structural similarity; close to -1 low.
    Boundary pixels are handled by reflect-padding before convolution so the
    spatial resolution is preserved exactly.
    """
    B, C, H, W = pred.shape
    kernel = _gaussian_kernel(window_size).to(pred.device, pred.dtype)
    kernel = kernel.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    # Treat all channels independently via groups=C
    mu1 = F.conv2d(pred,    kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target,  kernel, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=pad, groups=C) - mu12

    # Wang et al. (2004) stability constants for data range [0, 1]
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    num   = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
    # 1e-8 is invisible at bfloat16 precision (~3.5e-6 at this scale).
    # Use 1e-4 so the guard survives low-precision autocast on Ampere/Hopper.
    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-4
    return num / denom  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Cloud-aware loss module
# ---------------------------------------------------------------------------

class CloudAwareLoss(nn.Module):
    """Adaptive cloud-weighted multi-component reconstruction loss.

    Args:
        alpha:       Weight ratio for cloud vs clear pixels.
                     cloud → alpha, clear → (1-alpha). Default 0.8.
        lambda_mse:  Scale factor for the pixel-wise L2 term. Default 0.5.
        lambda_ssim: Scale factor for the 1-SSIM structural term. Default 0.5.
    """

    def __init__(
        self,
        alpha: float = 0.8,
        lambda_mse: float = 0.5,
        lambda_ssim: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim

    # ------------------------------------------------------------------
    # Weight map
    # ------------------------------------------------------------------

    def compute_weight_map(
        self,
        cloud_mask: torch.Tensor,
        cloud_thickness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build per-pixel loss weight map.

        Binary mask formula:
            W(x,y) = alpha * M'(x,y) + (1 − alpha) * (1 − M'(x,y))

        Thickness-weighted formula (when cloud_thickness is given):
            W(x,y) = alpha * thickness(x,y) + (1 − alpha) * (1 − M'(x,y))

        This ensures:
        - Thick cloud pixels receive ~80% of gradient weight (with alpha=0.8).
        - Clear pixels receive ~20% — enough to prevent background drift without
          dominating the loss.
        - Partially transparent clouds (thickness < 1) get proportionally less
          focus, reflecting that their reconstruction is easier.

        Args:
            cloud_mask:      (B, 1, H, W) binary mask, 1 = cloud, 0 = clear.
            cloud_thickness: (B, 1, H, W) continuous map in [0, 1]. Optional.
                             If None the binary mask is used.

        Returns:
            Weight map (B, 1, H, W) with values in [0, 1].
        """
        if cloud_thickness is not None:
            cloud_term = self.alpha * cloud_thickness
        else:
            cloud_term = self.alpha * cloud_mask

        clear_term = (1.0 - self.alpha) * (1.0 - cloud_mask)
        return cloud_term + clear_term   # (B, 1, H, W)

    # ------------------------------------------------------------------
    # SSIM
    # ------------------------------------------------------------------

    def ssim_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """1 − SSIM as a differentiable per-pixel loss map.

        Args:
            pred:   (B, C, H, W) predicted image.
            target: (B, C, H, W) ground-truth image.

        Returns:
            Per-pixel loss map (B, C, H, W) in [0, 2].
            Values near 0 = structurally identical; near 2 = fully dissimilar.
        """
        return 1.0 - _ssim_map(pred, target)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cloud_mask: Optional[torch.Tensor] = None,
        cloud_thickness: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the cloud-aware loss.

        L_final = mean( W * (lambda_mse * L_MSE + lambda_ssim * L_SSIM) )

        Where:
            L_MSE   = |pred − target|²        (per-pixel)
            L_SSIM  = 1 − SSIM(pred, target)  (per-pixel structural)
            W       = compute_weight_map(cloud_mask, cloud_thickness)

        Args:
            pred:            Reconstructed image (B, C, H, W).
            target:          Ground-truth clear image (B, C, H, W).
            cloud_mask:      Optional binary cloud mask (B, 1, H, W).
                             If None, uniform weights are used (standard loss).
            cloud_thickness: Optional continuous thickness map (B, 1, H, W).

        Returns:
            total_loss: Scalar differentiable loss tensor.
            loss_dict:  {mse, ssim, mse_weighted, ssim_weighted, total} — floats
                        detached from the graph, safe to log.
        """
        # --- Weight map (B, 1, H, W) ---
        if cloud_mask is not None:
            w = self.compute_weight_map(cloud_mask, cloud_thickness)
        else:
            w = torch.ones(
                pred.shape[0], 1, pred.shape[2], pred.shape[3],
                device=pred.device, dtype=pred.dtype,
            )

        # --- Per-pixel component maps ---
        mse_map  = (pred - target) ** 2          # (B, C, H, W)
        ssim_map = self.ssim_loss(pred, target)   # (B, C, H, W)

        # --- Weighted scalar loss ---
        combined  = self.lambda_mse * mse_map + self.lambda_ssim * ssim_map
        total     = (w * combined).mean()

        # --- Logging dict (unweighted scalars included for diagnostics) ---
        loss_dict = {
            "mse":           mse_map.mean().item(),
            "ssim":          ssim_map.mean().item(),
            "mse_weighted":  (w * mse_map).mean().item(),
            "ssim_weighted": (w * ssim_map).mean().item(),
            "total":         total.item(),
        }
        return total, loss_dict

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"lambda_mse={self.lambda_mse}, "
            f"lambda_ssim={self.lambda_ssim})"
        )


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def visualize_weight_map(
    weight_map: torch.Tensor,
    image: Optional[torch.Tensor] = None,
    cloud_mask: Optional[torch.Tensor] = None,
    sample_idx: int = 0,
    output_path: Optional[str] = None,
) -> None:
    """Plot the weight map overlaid on the image.

    Produces a figure with up to four panels:
        1. RGB image (first 3 channels, min-max normalised)  [if image given]
        2. Binary cloud mask                                  [if mask given]
        3. Weight map (hot colormap, annotated with min/max)
        4. 50%-opacity weight overlay on the image

    Args:
        weight_map:  (B, 1, H, W) weight map from CloudAwareLoss.compute_weight_map().
        image:       (B, C, H, W) image; first 3 channels used as RGB. Optional.
        cloud_mask:  (B, 1, H, W) binary cloud mask for reference. Optional.
        sample_idx:  Which batch element to visualise. Default 0.
        output_path: Save path (e.g. "weight_map.png"). plt.show() if None.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    # ---- Prepare numpy arrays ----
    w_np = weight_map[sample_idx, 0].detach().cpu().float().numpy()  # (H, W)

    img_np: Optional[np.ndarray] = None
    if image is not None:
        raw = image[sample_idx].detach().cpu().float()
        raw = raw[:3].permute(1, 2, 0).numpy()           # (H, W, 3)
        lo, hi = raw.min(), raw.max()
        img_np = np.clip((raw - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    mask_np: Optional[np.ndarray] = None
    if cloud_mask is not None:
        mask_np = cloud_mask[sample_idx, 0].detach().cpu().float().numpy()

    # ---- Layout ----
    panels = []
    if img_np  is not None: panels.append(("image", img_np))
    if mask_np is not None: panels.append(("mask",  mask_np))
    panels.append(("weight", w_np))
    panels.append(("overlay", None))   # always last
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (kind, data) in zip(axes, panels):
        if kind == "image":
            ax.imshow(data)
            ax.set_title("Image (RGB)")

        elif kind == "mask":
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)
            ax.set_title("Cloud Mask")

        elif kind == "weight":
            im = ax.imshow(data, cmap="hot", vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cloud_mean = data[data >= 0.5].mean() if (data >= 0.5).any() else float("nan")
            clear_mean = data[data <  0.5].mean() if (data <  0.5).any() else float("nan")
            ax.set_title(
                f"Weight Map\n"
                f"cloud avg={cloud_mean:.2f}  clear avg={clear_mean:.2f}"
            )

        elif kind == "overlay":
            # Background: image if available, else black
            bg = img_np if img_np is not None else np.zeros((*w_np.shape, 3))
            ax.imshow(bg)
            rgba = cm.hot(w_np)                  # (H, W, 4)
            rgba = rgba.copy()
            rgba[..., 3] = 0.55                  # 55% opacity for visibility
            ax.imshow(rgba)
            ax.set_title("Weight Overlay")

        ax.axis("off")

    plt.suptitle(
        "CloudAwareLoss — Weight Map Visualisation",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[CloudAwareLoss] Weight map saved → {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    torch.manual_seed(0)
    B, C, H, W = 2, 4, 64, 64

    pred   = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)
    mask   = (torch.rand(B, 1, H, W) > 0.5).float()
    thick  = torch.rand(B, 1, H, W)

    loss_fn = CloudAwareLoss(alpha=0.8, lambda_mse=0.5, lambda_ssim=0.5)
    print(repr(loss_fn))

    # ---- Weight map: binary ----
    w_bin = loss_fn.compute_weight_map(mask)
    assert w_bin.shape == (B, 1, H, W), f"Bad shape: {w_bin.shape}"
    cloud_w = w_bin[mask.bool()].mean().item()
    clear_w = w_bin[(1.0 - mask).bool()].mean().item()
    print(f"Binary weight — cloud: {cloud_w:.3f} (expect 0.800)  "
          f"clear: {clear_w:.3f} (expect 0.200)")
    assert abs(cloud_w - 0.8) < 1e-4, f"cloud weight wrong: {cloud_w}"
    assert abs(clear_w - 0.2) < 1e-4, f"clear weight wrong: {clear_w}"

    # ---- Weight map: thickness ----
    w_thick = loss_fn.compute_weight_map(mask, thick)
    assert w_thick.shape == (B, 1, H, W)
    print(f"Thickness weight range: [{w_thick.min():.3f}, {w_thick.max():.3f}]")

    # ---- Forward: no mask (uniform) ----
    loss, d = loss_fn(pred, target)
    print(f"No mask   — total={d['total']:.4f}  mse={d['mse']:.4f}  ssim={d['ssim']:.4f}")

    # ---- Forward: binary mask ----
    loss, d = loss_fn(pred, target, mask)
    print(f"Binary    — total={d['total']:.4f}  mse_w={d['mse_weighted']:.4f}  "
          f"ssim_w={d['ssim_weighted']:.4f}")

    # ---- Forward: thickness ----
    loss, d = loss_fn(pred, target, mask, thick)
    print(f"Thickness — total={d['total']:.4f}  mse_w={d['mse_weighted']:.4f}")

    # ---- Gradient check ----
    pred_g = pred.clone().requires_grad_(True)
    loss_g, _ = loss_fn(pred_g, target, mask)
    loss_g.backward()
    assert pred_g.grad is not None, "No gradient!"
    print(f"Gradient check — grad norm: {pred_g.grad.norm().item():.4f}")

    # ---- SSIM loss method ----
    ssim_map = loss_fn.ssim_loss(pred, target)
    assert ssim_map.shape == (B, C, H, W)
    print(f"SSIM map range: [{ssim_map.min().item():.3f}, {ssim_map.max().item():.3f}]")

    # ---- Degenerate: pred == target → SSIM loss ≈ 0, MSE = 0 ----
    loss_zero, d_zero = loss_fn(pred, pred, mask)
    assert d_zero["mse"] < 1e-8, f"MSE should be 0 for pred==target, got {d_zero['mse']}"
    print(f"pred==target — mse={d_zero['mse']:.2e}  ssim={d_zero['ssim']:.4f}  "
          f"total={d_zero['total']:.4f}")

    # ---- Visualisation ----
    visualize_weight_map(w_bin, pred, mask, output_path="weight_map_visualization.png")

    print("\nAll smoke tests passed.")
