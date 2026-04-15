"""Publication-quality visualisations for SAR-guided cloud removal.

All tensor inputs are assumed to be in [0, 1] float32 (pixel-space model).
SAR inputs are (2, H, W) with VV=channel 0, VH=channel 1.
Optical inputs use band ordering [B2, B3, B4, B8] (indices 0-3), so
natural-colour RGB is bands (2, 1, 0) = B4, B3, B2.

Functions
---------
plot_comparison              5-panel image comparison with metric annotations
plot_cloud_coverage_analysis Bar chart of PSNR/SSIM by cloud-coverage stratum
plot_nfe_tradeoff            Distortion–perception curve across NFE values
plot_ablation_table          Grouped bars comparing ablation study variants
plot_training_curves         Loss and validation metric curves over training

Legacy (backward-compat)
------------------------
save_comparison_grid         3-panel comparison (older API)
plot_band                    Single spectral band
plot_metric_curves           Generic dict-of-lists line plot
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker
    from matplotlib.patches import FancyArrowPatch
    _MPL = True
except ImportError:
    _MPL = False

# ---------------------------------------------------------------------------
# Paper-style rcParams
# ---------------------------------------------------------------------------

_PAPER_RC: Dict[str, Any] = {
    # Resolution
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
    # Fonts
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.size":           8,
    "axes.labelsize":      8,
    "axes.titlesize":      8.5,
    "axes.titleweight":    "bold",
    "xtick.labelsize":     7.5,
    "ytick.labelsize":     7.5,
    "legend.fontsize":     7.5,
    "legend.framealpha":   0.85,
    "legend.edgecolor":    "#cccccc",
    # Axes
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.8,
    "axes.labelpad":       4,
    # Grid
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.color":          "#b0b0b0",
    "grid.linewidth":      0.5,
    "grid.linestyle":      "--",
    # Lines / markers
    "lines.linewidth":     1.6,
    "lines.markersize":    5,
    # Background
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    # Layout
    "figure.constrained_layout.use": False,
}

# Colour palette — consistent across all figures
_C = {
    "blue":    "#2E86AB",   # our model / primary metric
    "orange":  "#E07B39",   # second variant
    "green":   "#3BB273",   # third variant
    "purple":  "#7B5EA7",   # fourth variant
    "red":     "#C94040",   # warning / bad
    "gray":    "#6B7280",   # reference / baseline
    "light":   "#D1D5DB",   # background bars
}

# Cloud-coverage bin labels (DB-CR Table III order)
_COV_BINS   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
_COV_LABELS = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]


@contextlib.contextmanager
def _paper_style():
    """Context manager: apply paper-quality rcParams temporarily."""
    if not _MPL:
        yield
        return
    with matplotlib.rc_context(_PAPER_RC):
        yield


# ---------------------------------------------------------------------------
# Low-level image helpers
# ---------------------------------------------------------------------------

def _to_display_rgb(
    tensor: torch.Tensor,
    rgb_idx: Tuple[int, int, int] = (2, 1, 0),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> np.ndarray:
    """Convert a (C, H, W) [0, 1] tensor to a (H, W, 3) float32 array for imshow.

    If vmin/vmax are given they override percentile stretching, enabling a
    consistent colour scale across multiple panels.
    """
    img = tensor.detach().cpu().float()
    rgb = torch.stack([img[b] for b in rgb_idx], dim=0)  # (3, H, W)
    arr = rgb.permute(1, 2, 0).numpy().astype(np.float32)  # (H, W, 3)
    if vmin is None:
        vmin = float(np.percentile(arr, p_lo))
    if vmax is None:
        vmax = float(np.percentile(arr, p_hi))
    return np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)


def _sar_to_display(sar: torch.Tensor) -> np.ndarray:
    """(2, H, W) SAR → (H, W, 3) false-colour float32 array.

    False-colour mapping: R=VV, G=mean(VV,VH), B=VH.
    Each channel is independently percentile-stretched for visibility.
    """
    vv  = sar[0].detach().cpu().float().numpy()
    vh  = sar[1].detach().cpu().float().numpy()
    avg = (vv + vh) / 2.0

    def _stretch(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    return np.stack([_stretch(vv), _stretch(avg), _stretch(vh)], axis=-1)


def _shared_vrange(
    *tensors: torch.Tensor,
    rgb_idx: Tuple[int, int, int] = (2, 1, 0),
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> Tuple[float, float]:
    """Compute a single (vmin, vmax) that covers all supplied tensors.

    Ensures the colour scale is identical across cloudy / predicted / GT panels
    so brightness differences are perceptually meaningful.
    """
    all_vals = []
    for t in tensors:
        img = t.detach().cpu().float()
        rgb = torch.stack([img[b] for b in rgb_idx], dim=0).permute(1, 2, 0).numpy()
        all_vals.append(rgb.ravel())
    combined = np.concatenate(all_vals)
    return float(np.percentile(combined, p_lo)), float(np.percentile(combined, p_hi))


def _save(fig: "plt.Figure", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. plot_comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    sar:          torch.Tensor,
    cloudy:       torch.Tensor,
    predicted:    torch.Tensor,
    ground_truth: torch.Tensor,
    cloud_mask:   Optional[torch.Tensor],
    save_path:    Union[str, Path],
    *,
    psnr_val:     Optional[float] = None,
    ssim_val:     Optional[float] = None,
    rgb_idx:      Tuple[int, int, int] = (2, 1, 0),
    title:        str = "",
) -> None:
    """Five-panel comparison figure: SAR | Cloudy | Predicted | Ground Truth | Cloud Mask.

    Cloudy, Predicted, and Ground Truth share the same percentile-stretch colour
    scale so brightness differences are directly comparable.  The predicted panel
    carries PSNR/SSIM annotations; if not supplied they are computed internally.

    Args:
        sar:          (2, H, W) Sentinel-1 VV/VH in [0, 1].
        cloudy:       (C, H, W) cloud-contaminated optical in [0, 1].
        predicted:    (C, H, W) bridge output in [0, 1].
        ground_truth: (C, H, W) cloud-free reference in [0, 1].
        cloud_mask:   Optional (1, H, W) binary mask {0, 1}.
        save_path:    Output file path (.png recommended).
        psnr_val:     Pre-computed PSNR in dB.  Computed if None.
        ssim_val:     Pre-computed SSIM in [0, 1].  Computed if None.
        rgb_idx:      Channel indices for (R, G, B).  Default = (B4, B3, B2).
        title:        Optional super-title string.
    """
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    # --- Compute metrics if not supplied ---
    if psnr_val is None or ssim_val is None:
        from utils.metrics import psnr as _psnr, ssim as _ssim
        p = predicted.unsqueeze(0).float()
        g = ground_truth.unsqueeze(0).float()
        if psnr_val is None:
            psnr_val = _psnr(p, g)
        if ssim_val is None:
            ssim_val = _ssim(p, g)

    # --- Shared stretch for optical panels ---
    vmin, vmax = _shared_vrange(cloudy, predicted, ground_truth, rgb_idx=rgb_idx)

    sar_img    = _sar_to_display(sar)
    cloudy_img = _to_display_rgb(cloudy,       rgb_idx, vmin, vmax)
    pred_img   = _to_display_rgb(predicted,    rgb_idx, vmin, vmax)
    gt_img     = _to_display_rgb(ground_truth, rgb_idx, vmin, vmax)

    n_panels = 5  # always show all 5; mask panel shows "no mask" if None

    with _paper_style():
        # Image panels wider than the mask panel
        w_ratios = [1.1, 1.0, 1.0, 1.0, 0.7]
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(3.3 * n_panels, 3.3),
            gridspec_kw={"width_ratios": w_ratios, "wspace": 0.06},
        )

        _titles = ["SAR (VV/VH)", "Cloudy Input", "Predicted", "Ground Truth", "Cloud Mask"]
        _imgs   = [sar_img, cloudy_img, pred_img, gt_img]

        for ax, img, ttl in zip(axes[:4], _imgs, _titles[:4]):
            ax.imshow(img, interpolation="bilinear")
            ax.set_title(ttl, pad=4)
            ax.axis("off")

        # --- Predicted panel: metric annotation ---
        axes[2].set_xlabel(
            f"PSNR {psnr_val:.2f} dB  ·  SSIM {ssim_val:.4f}",
            fontsize=7, color=_C["blue"], labelpad=3,
        )

        # --- Cloud Mask panel ---
        if cloud_mask is not None:
            mask_np = cloud_mask[0].detach().cpu().float().numpy()
            axes[4].imshow(mask_np, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
            cov = mask_np.mean()
            axes[4].set_xlabel(f"coverage {cov:.1%}", fontsize=7, labelpad=3)
        else:
            axes[4].text(
                0.5, 0.5, "no mask", ha="center", va="center",
                transform=axes[4].transAxes, fontsize=8, color=_C["gray"],
            )
        axes[4].set_title(_titles[4], pad=4)
        axes[4].axis("off")

        if title:
            fig.suptitle(title, fontsize=9, y=1.01, fontweight="bold")

        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# 2. plot_cloud_coverage_analysis
# ---------------------------------------------------------------------------

def plot_cloud_coverage_analysis(
    results_df,
    save_path: Union[str, Path],
    *,
    metrics:   Sequence[str] = ("psnr", "ssim"),
    title:     str = "Performance by Cloud Coverage (DB-CR Table III)",
) -> None:
    """Bar chart of metric means stratified by cloud-coverage bin.

    Args:
        results_df: pandas DataFrame **or** dict-of-lists with at minimum
                    ``cloud_coverage`` and metric columns.
                    ``cloud_coverage`` values are floats in [0, 1].
        save_path:  Output file path.
        metrics:    Which metrics to plot.  First metric uses the left y-axis;
                    second (if given) uses a secondary right y-axis.
        title:      Figure title.
    """
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    # --- Normalise input to dict of lists ---
    if hasattr(results_df, "to_dict"):     # pandas DataFrame
        data = {col: list(results_df[col]) for col in results_df.columns}
    else:
        data = dict(results_df)

    cov_arr = np.asarray(data["cloud_coverage"], dtype=np.float32)

    # Bin by coverage
    bin_indices = np.digitize(cov_arr, _COV_BINS[1:-1])  # 0-4
    bin_counts  = [int((bin_indices == i).sum()) for i in range(5)]

    # Per-bin means for each metric
    bin_means: Dict[str, List[float]] = {m: [] for m in metrics}
    bin_stds:  Dict[str, List[float]] = {m: [] for m in metrics}
    for i in range(5):
        mask = bin_indices == i
        for m in metrics:
            vals = np.asarray(data[m], dtype=np.float32)[mask]
            bin_means[m].append(float(vals.mean()) if len(vals) > 0 else float("nan"))
            bin_stds[m].append(float(vals.std())  if len(vals) > 1 else 0.0)

    x = np.arange(5)
    bar_colors = [
        _C["blue"], _C["orange"], _C["green"], _C["purple"], _C["gray"],
    ][:len(metrics)]

    with _paper_style():
        fig, ax1 = plt.subplots(figsize=(6.5, 3.6))

        # --- Primary metric (left axis) ---
        m0 = metrics[0]
        bars0 = ax1.bar(
            x - 0.2 if len(metrics) > 1 else x,
            bin_means[m0],
            width=0.38 if len(metrics) > 1 else 0.55,
            color=bar_colors[0],
            yerr=bin_stds[m0],
            capsize=3,
            error_kw={"linewidth": 0.8, "ecolor": "#666666"},
            label=m0.upper(),
            zorder=3,
        )
        ax1.set_ylabel(_metric_ylabel(m0), color=bar_colors[0])
        ax1.tick_params(axis="y", labelcolor=bar_colors[0])
        ax1.set_xticks(x)
        ax1.set_xticklabels(_COV_LABELS, rotation=0)
        ax1.set_xlabel("Cloud Coverage Bin")

        # Count annotation above each bar
        for xi, (bar, cnt) in enumerate(zip(bars0, bin_counts)):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                (bin_means[m0][xi] or 0) + (bin_stds[m0][xi] or 0) + 0.01 * ax1.get_ylim()[1],
                f"n={cnt}", ha="center", va="bottom",
                fontsize=6.5, color=_C["gray"],
            )

        # --- Secondary metric (right axis) ---
        if len(metrics) > 1:
            ax2   = ax1.twinx()
            m1    = metrics[1]
            ax2.bar(
                x + 0.2,
                bin_means[m1],
                width=0.38,
                color=bar_colors[1],
                alpha=0.85,
                yerr=bin_stds[m1],
                capsize=3,
                error_kw={"linewidth": 0.8, "ecolor": "#666666"},
                label=m1.upper(),
                zorder=3,
            )
            ax2.set_ylabel(_metric_ylabel(m1), color=bar_colors[1])
            ax2.tick_params(axis="y", labelcolor=bar_colors[1])
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_linewidth(0.8)

            # Combined legend
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2,
                       loc="upper right", frameon=True)

        ax1.set_title(title)
        ax1.set_axisbelow(True)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# 3. plot_nfe_tradeoff
# ---------------------------------------------------------------------------

def plot_nfe_tradeoff(
    nfe_results: Dict[int, Dict[str, float]],
    save_path:   Union[str, Path],
    *,
    default_nfe: int = 5,
    title:       str = "NFE Distortion–Perception Tradeoff (DB-CR Table V)",
) -> None:
    """Distortion (PSNR) vs Perception (LPIPS) scatter with NFE labels.

    Each point represents one NFE setting; the curve traces the Pareto frontier.
    Following DB-CR Table V, higher NFE improves both distortion and perception
    (unlike GAN-based methods where there is a hard tradeoff).

    Args:
        nfe_results: Dict mapping NFE (int) → metric dict with at least
                     ``psnr`` and optionally ``lpips``, ``ssim``.
                     Example::

                         {1: {"psnr": 30.1, "ssim": 0.891, "lpips": 0.178},
                          5: {"psnr": 33.9, "ssim": 0.923, "lpips": 0.139}}

        save_path:   Output file path.
        default_nfe: NFE value marked with a star (our recommended setting).
        title:       Figure title.
    """
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    nfes   = sorted(nfe_results)
    has_lpips = any("lpips" in nfe_results[n] for n in nfes)
    has_ssim  = any("ssim"  in nfe_results[n] for n in nfes)

    psnr_vals  = [nfe_results[n].get("psnr",  float("nan")) for n in nfes]
    lpips_vals = [nfe_results[n].get("lpips", float("nan")) for n in nfes]
    ssim_vals  = [nfe_results[n].get("ssim",  float("nan")) for n in nfes]

    n_subplots = 1 + (1 if has_lpips else 0) + (1 if has_ssim else 0)

    with _paper_style():
        fig, axes = plt.subplots(1, n_subplots, figsize=(3.5 * n_subplots, 3.4))
        if n_subplots == 1:
            axes = [axes]

        ax_idx = 0

        # --- Panel A: PSNR vs NFE ---
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(nfes, psnr_vals, "o-", color=_C["blue"], label="PSNR")
        for n, p in zip(nfes, psnr_vals):
            _annotate_point(ax, n, p, label=str(n), offset=(0, 5),
                            star=(n == default_nfe), color=_C["blue"])
        ax.set_xlabel("NFE (inference steps)")
        ax.set_ylabel("PSNR (dB) ↑")
        ax.set_xticks(nfes)
        ax.set_title("Distortion")
        ax.grid(True, alpha=0.25)

        # --- Panel B: SSIM vs NFE ---
        if has_ssim:
            ax = axes[ax_idx]; ax_idx += 1
            ax.plot(nfes, ssim_vals, "s-", color=_C["green"], label="SSIM")
            for n, s in zip(nfes, ssim_vals):
                _annotate_point(ax, n, s, label=str(n), offset=(0, 3e-3),
                                star=(n == default_nfe), color=_C["green"])
            ax.set_xlabel("NFE (inference steps)")
            ax.set_ylabel("SSIM ↑")
            ax.set_xticks(nfes)
            ax.set_title("Structural Similarity")
            ax.grid(True, alpha=0.25)

        # --- Panel C: PSNR vs LPIPS scatter ---
        if has_lpips:
            ax = axes[ax_idx]; ax_idx += 1
            ax.plot(psnr_vals, lpips_vals, "D--", color=_C["gray"],
                    linewidth=1.0, zorder=1)
            sc = ax.scatter(psnr_vals, lpips_vals,
                            c=nfes, cmap="Blues", s=50,
                            zorder=3, edgecolors="#333333", linewidths=0.5)
            for n, p, l in zip(nfes, psnr_vals, lpips_vals):
                _annotate_point(ax, p, l, label=f"NFE={n}", offset=(0.1, 0.003),
                                star=(n == default_nfe), color=_C["blue"])
            plt.colorbar(sc, ax=ax, label="NFE", fraction=0.046, pad=0.04)
            ax.set_xlabel("PSNR (dB) ↑")
            ax.set_ylabel("LPIPS ↓")
            ax.set_title("Distortion vs Perception")
            # "Better" arrow (top-right = lower LPIPS, higher PSNR)
            ax.annotate(
                "better →",
                xy=(0.72, 0.08), xycoords="axes fraction",
                fontsize=7, color=_C["green"],
                ha="left",
            )
            ax.grid(True, alpha=0.25)

        fig.suptitle(title, fontsize=8.5, y=1.01)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# 4. plot_ablation_table
# ---------------------------------------------------------------------------

def plot_ablation_table(
    ablation_results: Dict[str, Dict[str, float]],
    save_path:        Union[str, Path],
    *,
    metrics:          Sequence[str] = ("psnr", "ssim", "mae", "sam"),
    baseline_key:     Optional[str] = None,
    title:            str = "Ablation Study",
) -> None:
    """Grouped bar chart comparing ablation study variants across metrics.

    Args:
        ablation_results: Dict mapping variant name → metric dict.
                          Example::

                              {
                                  "Ours (full)":           {"psnr": 33.9, "ssim": 0.923},
                                  "w/o VimSSM":            {"psnr": 32.1, "ssim": 0.908},
                                  "w/o SFBlock":           {"psnr": 31.5, "ssim": 0.897},
                              }

        save_path:        Output file path.
        metrics:          Metrics to display (up to 4; one subplot each).
        baseline_key:     Variant name of the baseline (gets a dashed reference
                          line in each subplot).  Defaults to the first key.
        title:            Figure super-title.
    """
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    variants = list(ablation_results)
    if baseline_key is None:
        baseline_key = variants[0]

    metrics = [m for m in metrics if any(m in v for v in ablation_results.values())][:4]
    n_metrics = len(metrics)
    n_cols    = min(n_metrics, 2)
    n_rows    = math.ceil(n_metrics / n_cols)

    colors = [_C["blue"], _C["orange"], _C["green"], _C["purple"], _C["gray"]]
    bar_colors = {v: colors[i % len(colors)] for i, v in enumerate(variants)}
    # Baseline gets a muted shade
    if baseline_key in bar_colors:
        bar_colors[baseline_key] = _C["gray"]

    x = np.arange(len(variants))
    bar_w = 0.55

    with _paper_style():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.5 * n_cols, 3.0 * n_rows),
            squeeze=False,
        )

        for k, metric in enumerate(metrics):
            ax  = axes[k // n_cols][k % n_cols]
            vals = [ablation_results[v].get(metric, float("nan")) for v in variants]
            base_val = ablation_results[baseline_key].get(metric, float("nan"))

            bars = ax.bar(
                x, vals, width=bar_w,
                color=[bar_colors[v] for v in variants],
                edgecolor="white", linewidth=0.5,
                zorder=3,
            )

            # Dashed baseline reference line
            if not math.isnan(base_val):
                ax.axhline(base_val, color=_C["gray"], linewidth=1.0,
                           linestyle="--", zorder=2, alpha=0.7)

            # Value labels on top of each bar
            for bar, val in zip(bars, vals):
                if not math.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.005 * abs(bar.get_height()),
                        _fmt_metric(metric, val),
                        ha="center", va="bottom", fontsize=6.5,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [_wrap_label(v, 14) for v in variants],
                rotation=25, ha="right", fontsize=7,
            )
            ax.set_ylabel(_metric_ylabel(metric))
            ax.set_title(_metric_full_name(metric))
            ax.set_axisbelow(True)

            # Y-axis range: give 10% headroom above tallest bar
            finite = [v for v in vals if not math.isnan(v)]
            if finite:
                span = max(finite) - min(finite)
                pad  = max(span * 0.2, abs(max(finite)) * 0.05)
                ax.set_ylim(min(finite) - pad * 0.3, max(finite) + pad)

        # Hide any unused subplots
        for k in range(n_metrics, n_rows * n_cols):
            axes[k // n_cols][k % n_cols].set_visible(False)

        fig.suptitle(title, fontsize=9, y=1.02)
        plt.tight_layout(pad=1.2)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# 5. plot_training_curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history:   Union[Dict[str, List[float]], Any],
    save_path: Union[str, Path],
    *,
    title:     str = "Training History",
    smooth:    float = 0.0,
) -> None:
    """Four-panel training curve figure from W&B run history or a dict.

    Layout::

        ┌──────────────────┬──────────────────┐
        │  Training Loss   │  Validation PSNR │
        ├──────────────────┼──────────────────┤
        │  Validation SSIM │  Learning Rate   │
        └──────────────────┴──────────────────┘

    Args:
        history:   Dict mapping metric name → list of scalar values (one per
                   logged step/epoch).  Also accepts a pandas DataFrame
                   (columns = metric names, rows = steps).

                   Expected keys (all optional; present subset is plotted)::

                       "train/total"   or  "train_loss"
                       "train/mse"
                       "train/ssim"
                       "val/psnr"      or  "val_psnr"
                       "val/ssim"      or  "val_ssim"
                       "val/mae"       or  "val_mae"
                       "train/lr"      or  "lr"

        save_path: Output file path.
        title:     Figure super-title.
        smooth:    Exponential moving average coefficient for loss curves.
                   0.0 = no smoothing; 0.9 = heavy smoothing.
    """
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    # --- Normalise to dict ---
    if hasattr(history, "to_dict"):        # pandas DataFrame
        hist: Dict[str, List[float]] = {
            col: list(history[col].dropna()) for col in history.columns
        }
    else:
        hist = {k: list(v) for k, v in history.items()}

    def _get(*keys: str) -> Optional[List[float]]:
        for k in keys:
            if k in hist and hist[k]:
                return hist[k]
        return None

    def _ema(vals: List[float], alpha: float) -> List[float]:
        if alpha <= 0:
            return vals
        out, s = [], vals[0]
        for v in vals:
            s = alpha * s + (1 - alpha) * v
            out.append(s)
        return out

    train_loss = _get("train/total", "train_loss", "loss/train")
    train_mse  = _get("train/mse",   "train/mse_weighted")
    train_ssim = _get("train/ssim",  "train/ssim_weighted")
    val_psnr   = _get("val/psnr",    "val_psnr")
    val_ssim   = _get("val/ssim",    "val_ssim")
    val_mae    = _get("val/mae",     "val_mae")
    lr_vals    = _get("train/lr",    "lr", "learning_rate")

    with _paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
        ((ax_loss, ax_psnr), (ax_ssim, ax_lr)) = axes

        # --- Panel 1: Training Loss ---
        plotted_loss = False
        if train_loss:
            steps = np.arange(len(train_loss))
            ax_loss.plot(steps, _ema(train_loss, smooth),
                         color=_C["blue"], label="total", linewidth=1.6)
            plotted_loss = True
        if train_mse:
            steps = np.arange(len(train_mse))
            ax_loss.plot(steps, _ema(train_mse, smooth),
                         color=_C["orange"], label="MSE (weighted)",
                         linewidth=1.2, linestyle="--")
            plotted_loss = True
        if train_ssim:
            steps = np.arange(len(train_ssim))
            ax_loss.plot(steps, _ema(train_ssim, smooth),
                         color=_C["green"], label="SSIM (weighted)",
                         linewidth=1.2, linestyle=":")
            plotted_loss = True
        if plotted_loss:
            ax_loss.set_xlabel("Training Step")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("Training Loss")
            ax_loss.legend(loc="upper right")
        else:
            ax_loss.text(0.5, 0.5, "no loss data", ha="center", va="center",
                         transform=ax_loss.transAxes, color=_C["gray"])
            ax_loss.set_title("Training Loss")

        # --- Panel 2: Validation PSNR ---
        _plot_val_curve(ax_psnr, val_psnr, "val/psnr", "PSNR (dB) ↑",
                        "Validation PSNR", _C["blue"])

        # --- Panel 3: Validation SSIM ---
        _plot_val_curve(ax_ssim, val_ssim, "val/ssim", "SSIM ↑",
                        "Validation SSIM", _C["green"])

        # Add MAE as secondary if available
        if val_mae:
            ax_r = ax_ssim.twinx()
            ax_r.plot(np.arange(len(val_mae)), val_mae, color=_C["orange"],
                      linewidth=1.1, linestyle="--", label="MAE ↓", alpha=0.8)
            ax_r.set_ylabel("MAE ↓", color=_C["orange"], fontsize=7.5)
            ax_r.tick_params(axis="y", labelcolor=_C["orange"], labelsize=7)
            ax_r.spines["right"].set_visible(True)
            ax_r.spines["right"].set_linewidth(0.6)

        # --- Panel 4: Learning Rate ---
        if lr_vals:
            steps = np.arange(len(lr_vals))
            ax_lr.semilogy(steps, lr_vals, color=_C["purple"], linewidth=1.4)
            ax_lr.set_xlabel("Training Step")
            ax_lr.set_ylabel("Learning Rate")
            ax_lr.set_title("LR Schedule")
            ax_lr.yaxis.set_minor_formatter(mticker.NullFormatter())
        else:
            ax_lr.text(0.5, 0.5, "no LR data", ha="center", va="center",
                       transform=ax_lr.transAxes, color=_C["gray"])
            ax_lr.set_title("LR Schedule")

        fig.suptitle(title, fontsize=9.5, y=1.01, fontweight="bold")
        plt.tight_layout(pad=1.5)
        _save(fig, Path(save_path))


# ---------------------------------------------------------------------------
# Internal helpers for plots
# ---------------------------------------------------------------------------

def _annotate_point(
    ax:     "plt.Axes",
    x:      float,
    y:      float,
    label:  str,
    offset: Tuple[float, float] = (0, 5),
    star:   bool = False,
    color:  str = "#333333",
) -> None:
    """Annotate a scatter point with a text label; optionally mark with ★."""
    ax.annotate(
        ("★ " if star else "") + label,
        xy=(x, y), xytext=(x + offset[0], y + offset[1]),
        fontsize=7, color=color, ha="center", va="bottom",
    )


def _plot_val_curve(
    ax:     "plt.Axes",
    vals:   Optional[List[float]],
    key:    str,
    ylabel: str,
    title:  str,
    color:  str,
) -> None:
    if vals:
        epochs = np.arange(len(vals))
        ax.plot(epochs, vals, "o-", color=color, markersize=3, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
    else:
        ax.text(0.5, 0.5, f"no {key} data", ha="center", va="center",
                transform=ax.transAxes, color=_C["gray"])
    ax.set_title(title)


def _metric_ylabel(metric: str) -> str:
    mapping = {
        "psnr":        "PSNR (dB) ↑",
        "ssim":        "SSIM ↑",
        "mae":         "MAE ↓",
        "sam":         "SAM (°) ↓",
        "lpips":       "LPIPS ↓",
        "psnr_cloud":  "Cloud PSNR (dB) ↑",
        "mae_cloud":   "Cloud MAE ↓",
    }
    return mapping.get(metric.lower(), metric)


def _metric_full_name(metric: str) -> str:
    mapping = {
        "psnr":        "Peak SNR",
        "ssim":        "Structural Similarity",
        "mae":         "Mean Abs. Error",
        "sam":         "Spectral Angle",
        "lpips":       "Perceptual (LPIPS)",
        "psnr_cloud":  "Cloud-region PSNR",
        "mae_cloud":   "Cloud-region MAE",
    }
    return mapping.get(metric.lower(), metric.upper())


def _fmt_metric(metric: str, val: float) -> str:
    fmt = {
        "psnr": "{:.1f}", "psnr_cloud": "{:.1f}",
        "ssim": "{:.3f}",
        "mae":  "{:.4f}", "mae_cloud":  "{:.4f}",
        "sam":  "{:.2f}",
        "lpips":"{:.3f}",
    }
    return fmt.get(metric.lower(), "{:.3f}").format(val)


def _wrap_label(label: str, width: int) -> str:
    """Soft-wrap a long label at word boundaries."""
    words, lines, line = label.split(), [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy API  (kept for backward-compat)
# ---------------------------------------------------------------------------

def save_comparison_grid(
    cloudy:        torch.Tensor,
    reconstructed: torch.Tensor,
    clear:         torch.Tensor,
    save_path:     Union[str, Path],
    cloud_mask:    Optional[torch.Tensor] = None,
    rgb_bands:     Tuple[int, int, int] = (2, 1, 0),
    title:         str = "",
) -> None:
    """3-panel (or 4-panel with mask) comparison.  Prefer ``plot_comparison``."""
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    vmin, vmax = _shared_vrange(cloudy, reconstructed, clear, rgb_idx=rgb_bands)
    n = 4 if cloud_mask is not None else 3

    with _paper_style():
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
        for ax, img, lbl in zip(
            axes,
            [cloudy, reconstructed, clear],
            ["Cloudy", "Reconstructed", "Ground Truth"],
        ):
            ax.imshow(_to_display_rgb(img, rgb_bands, vmin, vmax))
            ax.set_title(lbl)
            ax.axis("off")

        if cloud_mask is not None:
            axes[-1].imshow(cloud_mask[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[-1].set_title("Cloud Mask")
            axes[-1].axis("off")

        if title:
            fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        _save(fig, Path(save_path))


def plot_band(
    tensor:    torch.Tensor,
    band_idx:  int,
    save_path: Union[str, Path],
    cmap:      str = "gray",
    title:     str = "",
) -> None:
    """Render a single spectral band as a greyscale image."""
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    arr = tensor[band_idx].detach().cpu().float().numpy()
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    with _paper_style():
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
        if title:
            ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save(fig, Path(save_path))


def plot_metric_curves(
    metric_history: Dict[str, List[float]],
    save_path:      Union[str, Path],
    title:          str = "Training Metrics",
) -> None:
    """Generic line plot: one curve per key in metric_history."""
    if not _MPL:
        raise ImportError("matplotlib is required: pip install matplotlib")

    colors = list(_C.values())
    with _paper_style():
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for i, (name, vals) in enumerate(metric_history.items()):
            ax.plot(vals, label=name, color=colors[i % len(colors)])
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend(ncol=min(len(metric_history), 3))
        _save(fig, Path(save_path))
