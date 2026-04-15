"""Evaluation script for the OT-ODE diffusion bridge.

Runs the trained model on the test split, computes all image quality metrics,
stratifies by cloud coverage percentage, saves per-sample CSV, and optionally
runs an NFE sweep to show the distortion-perception tradeoff.

Usage
-----
    # Evaluate at 5 NFE (default):
    python train/evaluate.py --ckpt outputs/checkpoints/bridge/best.ckpt

    # Evaluate at 1, 3, and 5 NFE:
    python train/evaluate.py --ckpt best.ckpt --nfe 1 3 5

    # Full sweep + visualizations every 20th sample:
    python train/evaluate.py --ckpt best.ckpt --sweep --viz-every 20

    # Override any config key:
    python train/evaluate.py --ckpt best.ckpt training.batch_size=4
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from omegaconf import OmegaConf
    _OMEGACONF = True
except ImportError:
    _OMEGACONF = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from data import SEN12MSCRDataset, collate_fn
from models import SAROpticalUNet, DiffusionBridge
from models.bridge.noise_schedule import BridgeNoiseSchedule
from models.cloud_aware_loss import CloudAwareLoss
from utils.metrics import (
    compute_metrics,
    per_sample_metrics,
    MetricAggregator,
)

log = logging.getLogger(__name__)

# DB-CR cloud coverage stratification bins (following the paper's evaluation protocol)
_COVERAGE_BINS: List[Tuple[float, float]] = [
    (0.0,  0.2),
    (0.2,  0.4),
    (0.4,  0.6),
    (0.6,  0.8),
    (0.8,  1.01),   # upper edge slightly > 1 to catch 100% coverage
]
_BIN_LABELS = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

# Metric display order and formatting
_METRIC_FMTS: Dict[str, Tuple[str, str, str]] = {
    # key: (display_name, format_str, direction)
    "psnr":       ("PSNR (dB)", "{:6.2f}",  "↑"),
    "ssim":       ("SSIM",      "{:6.4f}",  "↑"),
    "mae":        ("MAE",       "{:7.5f}",  "↓"),
    "sam":        ("SAM (°)",   "{:6.3f}",  "↓"),
    "lpips":      ("LPIPS",     "{:6.4f}",  "↓"),
    "psnr_cloud": ("PSNR-cld",  "{:6.2f}",  "↑"),
    "mae_cloud":  ("MAE-cld",   "{:7.5f}",  "↓"),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, cfg, device: torch.device) -> DiffusionBridge:
    """Build the bridge and load **EMA weights** from a training checkpoint.

    The training script saves both live weights (``"bridge"`` key) and EMA
    shadow weights (``"ema"`` key).  For evaluation we use the EMA copy,
    which is consistently better than the live weights near convergence.
    """
    unet = SAROpticalUNet(
        in_channels_optical=cfg.model.in_channels_optical,
        in_channels_sar=cfg.data.sar_bands,
        base_channels=cfg.model.base_channels,
        channel_mult=list(cfg.model.channel_mult),
        num_nafblocks=list(cfg.model.num_nafblocks),
        num_dec_nafblocks=list(cfg.model.num_dec_nafblocks),
        num_vim_blocks=cfg.model.num_vim_blocks,
        time_emb_dim=cfg.model.time_emb_dim,
        dropout=cfg.model.dropout,
        vim_d_state=cfg.model.vim_d_state,
    ).to(device)

    schedule = BridgeNoiseSchedule(
        num_steps=cfg.diffusion.diffusion_steps,
        schedule_type=cfg.diffusion.alpha_schedule_type,
    )
    criterion = CloudAwareLoss(
        alpha=cfg.loss.alpha,
        lambda_mse=cfg.loss.lambda_mse,
        lambda_ssim=cfg.loss.lambda_ssim,
    )
    bridge = DiffusionBridge(
        model=unet,
        noise_schedule=schedule,
        device=device,
        loss_fn=criterion,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Prefer EMA weights; fall back to live weights for forward-compat
    if "ema" in ckpt:
        model_state = bridge.state_dict()
        ema_state   = {
            k: v.to(dtype=model_state[k].dtype)
            for k, v in ckpt["ema"].items()
            if k in model_state
        }
        bridge.load_state_dict(ema_state, strict=False)
        log.info("Loaded EMA weights from %s  (epoch %s)", ckpt_path, ckpt.get("epoch", "?"))
    else:
        bridge.load_state_dict(ckpt["bridge"])
        log.info("Loaded live weights from %s  (epoch %s)", ckpt_path, ckpt.get("epoch", "?"))

    bridge.eval()
    return bridge


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _to_uint8(
    tensor: torch.Tensor,
    rgb_idx: Tuple[int, int, int] = (2, 1, 0),
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> np.ndarray:
    """Convert a (C, H, W) [0, 1] tensor to a (H, W, 3) uint8 RGB array.

    Uses percentile contrast stretching for display clarity.
    ``rgb_idx`` selects three channel indices as R, G, B respectively.
    Defaults to (B4, B3, B2) = true-colour for Sentinel-2 band order [1,2,3,7].
    """
    img = tensor.detach().cpu().float()
    rgb = torch.stack([img[b] for b in rgb_idx], dim=0)   # (3, H, W)
    arr = rgb.permute(1, 2, 0).numpy()                     # (H, W, 3)
    lo  = np.percentile(arr, p_lo)
    hi  = np.percentile(arr, p_hi)
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _sar_to_uint8(sar: torch.Tensor) -> np.ndarray:
    """Convert a (2, H, W) SAR tensor (VV=ch0, VH=ch1) → (H, W, 3) pseudo-colour.

    False-colour mapping: R=VV, G=mean(VV,VH), B=VH.  Bright areas in VV
    indicate double-bounce (urban); higher VH indicates volume scattering (veg).
    """
    vv  = sar[0].detach().cpu().float().numpy()
    vh  = sar[1].detach().cpu().float().numpy()
    avg = (vv + vh) / 2.0

    def _stretch(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    rgb = np.stack([_stretch(vv), _stretch(avg), _stretch(vh)], axis=-1)  # (H, W, 3)
    return (rgb * 255).astype(np.uint8)


def save_comparison_strip(
    sar:        torch.Tensor,
    cloudy:     torch.Tensor,
    pred:       torch.Tensor,
    clean:      torch.Tensor,
    save_path:  Path,
    cloud_mask: Optional[torch.Tensor] = None,
    psnr_val:   Optional[float]         = None,
    ssim_val:   Optional[float]         = None,
    sam_val:    Optional[float]         = None,
    coverage:   Optional[float]         = None,
) -> None:
    """Save a [SAR | Cloudy | Predicted | Ground Truth] comparison figure.

    Args:
        sar:        (2, H, W) Sentinel-1 VV/VH  [0, 1].
        cloudy:     (C, H, W) cloudy optical     [0, 1].
        pred:       (C, H, W) bridge output      [0, 1].
        clean:      (C, H, W) ground truth       [0, 1].
        save_path:  Output PNG path.
        cloud_mask: Optional (1, H, W) binary mask for overlay.
        psnr_val, ssim_val, sam_val: Optional metric annotations.
        coverage:   Cloud coverage fraction for figure title.
    """
    if not _MPL:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Select RGB bands: for [B2,B3,B4,B8] band order, true-colour is (2,1,0)
    rgb_idx = (2, 1, 0)

    sar_img    = _sar_to_uint8(sar)
    cloudy_img = _to_uint8(cloudy, rgb_idx)
    pred_img   = _to_uint8(pred,   rgb_idx)
    clean_img  = _to_uint8(clean,  rgb_idx)

    n_panels = 5 if cloud_mask is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5),
                             gridspec_kw={"wspace": 0.05})

    titles = ["SAR (VV/VH)", "Cloudy", "Predicted", "Ground Truth"]
    imgs   = [sar_img, cloudy_img, pred_img, clean_img]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    if cloud_mask is not None:
        mask_np = cloud_mask[0].detach().cpu().numpy()
        axes[-1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[-1].set_title("Cloud Mask", fontsize=9, fontweight="bold")
        axes[-1].axis("off")

    # Annotate predicted panel with metrics
    annotation_parts = []
    if psnr_val is not None:
        annotation_parts.append(f"PSNR={psnr_val:.2f} dB")
    if ssim_val is not None:
        annotation_parts.append(f"SSIM={ssim_val:.4f}")
    if sam_val is not None:
        annotation_parts.append(f"SAM={sam_val:.2f}°")
    if annotation_parts:
        axes[2].set_xlabel("  ".join(annotation_parts), fontsize=7.5, color="steelblue")

    cov_str = f"  |  cloud coverage: {coverage:.1%}" if coverage is not None else ""
    fig.suptitle(f"SAR-Guided Cloud Removal{cov_str}", fontsize=10)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stratification helpers
# ---------------------------------------------------------------------------

def _coverage_bin(coverage: float) -> int:
    """Return the bin index (0–4) for a given cloud coverage fraction."""
    for i, (lo, hi) in enumerate(_COVERAGE_BINS):
        if lo <= coverage < hi:
            return i
    return len(_COVERAGE_BINS) - 1   # catch-all for exactly 1.0


def stratify_records(
    records: List[dict],
) -> Dict[str, List[dict]]:
    """Group per-sample result dicts by cloud coverage bin label.

    Args:
        records: List of dicts, each containing a ``"cloud_coverage"`` float.

    Returns:
        Dict mapping bin label (e.g. ``"20–40%"``) to list of sample records.
    """
    bins: Dict[str, List[dict]] = {lbl: [] for lbl in _BIN_LABELS}
    for rec in records:
        lbl = _BIN_LABELS[_coverage_bin(float(rec.get("cloud_coverage", 0.5)))]
        bins[lbl].append(rec)
    return bins


def _bin_mean(records: List[dict], metric_keys: List[str]) -> Dict[str, float]:
    """Compute mean of selected keys across a list of record dicts."""
    if not records:
        return {k: float("nan") for k in metric_keys}
    agg = MetricAggregator()
    for r in records:
        agg.update({k: r[k] for k in metric_keys if k in r})
    return agg.compute()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(records: List[dict], path: Path) -> None:
    """Write one row per sample to a CSV file.

    The column order is: patch_id, nfe, cloud_coverage, season, then all
    metric keys sorted alphabetically.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    meta_keys    = ["patch_id", "nfe", "cloud_coverage", "season", "roi"]
    metric_keys  = sorted(k for k in records[0] if k not in meta_keys)
    fieldnames   = meta_keys + metric_keys

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    log.info("Per-sample CSV saved → %s  (%d rows)", path, len(records))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt_val(key: str, val: float) -> str:
    fmt = _METRIC_FMTS.get(key, (key, "{:.4f}", ""))[1]
    try:
        return fmt.format(val)
    except (ValueError, KeyError):
        return f"{val:.4f}"


def print_summary_table(
    global_means:   Dict[str, float],
    strat_means:    Dict[str, Dict[str, float]],
    metric_keys:    List[str],
    nfe:            int,
    total_samples:  int,
) -> None:
    """Print a stratified results table to stdout."""
    displayed   = [k for k in metric_keys if k in _METRIC_FMTS]
    if not displayed:
        displayed = metric_keys

    col_w = 10
    bin_w = 12

    # Header
    print()
    print(f"  Evaluation Results  |  NFE={nfe}  |  N={total_samples} samples")
    print("  " + "─" * (bin_w + len(displayed) * (col_w + 1) + 8))

    header = f"  {'Coverage':<{bin_w}}  {'N':>5}  "
    for k in displayed:
        direction = _METRIC_FMTS[k][2]
        label     = _METRIC_FMTS[k][0]
        header   += f"{label+direction:>{col_w}}  "
    print(header)
    print("  " + "─" * (bin_w + len(displayed) * (col_w + 1) + 8))

    # Stratified rows
    for lbl, means in strat_means.items():
        n   = means.pop("_count", 0)
        row = f"  {lbl:<{bin_w}}  {int(n):>5}  "
        for k in displayed:
            v = means.get(k, float("nan"))
            row += f"{_fmt_val(k, v):>{col_w}}  "
        print(row)

    print("  " + "─" * (bin_w + len(displayed) * (col_w + 1) + 8))

    # Global average row
    row = f"  {'Overall':<{bin_w}}  {total_samples:>5}  "
    for k in displayed:
        v = global_means.get(k, float("nan"))
        row += f"{_fmt_val(k, v):>{col_w}}  "
    print(row)
    print("  " + "─" * (bin_w + len(displayed) * (col_w + 1) + 8))
    print()


def print_nfe_sweep_table(
    sweep_results: Dict[int, Dict[str, float]],
    metric_keys:   List[str],
) -> None:
    """Print a compact NFE vs. metric tradeoff table."""
    displayed = [k for k in metric_keys if k in _METRIC_FMTS]
    col_w = 10

    print()
    print("  NFE Sweep — Distortion / Perception Tradeoff")
    print("  " + "─" * (6 + len(displayed) * (col_w + 1) + 2))

    header = f"  {'NFE':>4}  "
    for k in displayed:
        label, _, direction = _METRIC_FMTS[k]
        header += f"{label+direction:>{col_w}}  "
    print(header)
    print("  " + "─" * (6 + len(displayed) * (col_w + 1) + 2))

    for nfe, means in sorted(sweep_results.items()):
        row = f"  {nfe:>4}  "
        for k in displayed:
            v = means.get(k, float("nan"))
            row += f"{_fmt_val(k, v):>{col_w}}  "
        print(row)

    print("  " + "─" * (6 + len(displayed) * (col_w + 1) + 2))
    print()


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    bridge:    DiffusionBridge,
    loader:    DataLoader,
    nfe:       int,
    cfg,
    device:    torch.device,
    viz_dir:   Path,
    viz_every: int,
    use_lpips: bool,
    use_amp:   bool,
) -> Tuple[List[dict], Dict[str, float]]:
    """Run bridge.sample() over the full loader at ``nfe`` steps.

    Returns:
        records:      List of per-sample dicts (metrics + metadata + nfe).
        global_means: Mean of all metrics across the test set.
    """
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    records:  List[dict] = []
    global_agg = MetricAggregator()
    sample_idx = 0

    iterable = tqdm(loader, desc=f"NFE={nfe}", unit="batch") if _TQDM else loader

    for batch in iterable:
        x_cloudy   = batch["cloudy"].to(device, non_blocking=True)
        x_clean    = batch["clean"].to(device, non_blocking=True)
        sar        = batch["sar"].to(device, non_blocking=True)
        cloud_mask = batch.get("cloud_mask")
        if cloud_mask is not None:
            cloud_mask = cloud_mask.to(device, non_blocking=True)
        metadata: List[dict] = batch.get("metadata", [{} for _ in range(x_cloudy.shape[0])])

        # --- Inference ---
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            x_pred = bridge.sample(
                x_cloudy, sar,
                cloud_mask=cloud_mask,
                num_steps=nfe,
            )

        x_pred_f  = x_pred.float().clamp(0.0, 1.0)
        x_clean_f = x_clean.float().clamp(0.0, 1.0)

        # --- Per-sample metrics ---
        sample_metrics = per_sample_metrics(
            x_pred_f, x_clean_f, cloud_mask, use_lpips=use_lpips,
        )

        # --- Record results + metadata ---
        for i, m in enumerate(sample_metrics):
            meta = metadata[i] if i < len(metadata) else {}
            coverage = float(meta.get("cloud_coverage", 0.5))

            rec: dict = {
                "patch_id":       meta.get("patch_id", f"{sample_idx + i}"),
                "nfe":            nfe,
                "cloud_coverage": coverage,
                "season":         meta.get("season", ""),
                "roi":            meta.get("roi", ""),
                **m,
            }
            records.append(rec)
            global_agg.update(m)

            # --- Visualisation ---
            if viz_every > 0 and (sample_idx + i) % viz_every == 0:
                strip_path = viz_dir / f"nfe{nfe}" / f"sample_{sample_idx + i:05d}.png"
                save_comparison_strip(
                    sar=sar[i],
                    cloudy=x_cloudy[i],
                    pred=x_pred_f[i],
                    clean=x_clean_f[i],
                    save_path=strip_path,
                    cloud_mask=cloud_mask[i : i + 1] if cloud_mask is not None else None,
                    psnr_val=m.get("psnr"),
                    ssim_val=m.get("ssim"),
                    sam_val=m.get("sam"),
                    coverage=coverage,
                )

        sample_idx += x_cloudy.shape[0]

    return records, global_agg.compute()


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def evaluate(cfg, args: argparse.Namespace) -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    log.info("Evaluating on %s  (AMP=%s)", device, use_amp)

    # --- Dataset ---
    test_ds = SEN12MSCRDataset.from_config(cfg, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size or cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    log.info("Test set: %d samples, %d batches", len(test_ds), len(test_loader))

    # --- Model ---
    ckpt_path = Path(args.ckpt)
    bridge    = load_model(ckpt_path, cfg, device)

    # --- Output directories ---
    out_root = Path(args.out_dir) if args.out_dir else Path(cfg.paths.output_root) / "eval"
    viz_dir  = out_root / "visualizations"
    csv_dir  = out_root / "metrics"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Determine which NFE values to run
    nfe_list: List[int] = sorted(set(args.nfe)) if args.nfe else [cfg.diffusion.sampler_nfe]
    if args.sweep and 1 not in nfe_list:
        nfe_list = sorted({1, 3, 5} | set(nfe_list))

    sweep_results: Dict[int, Dict[str, float]] = {}

    # =====================================================================
    # Per-NFE evaluation loop
    # =====================================================================
    for nfe in nfe_list:
        log.info("Running inference at NFE=%d ...", nfe)

        records, global_means = run_inference(
            bridge=bridge,
            loader=test_loader,
            nfe=nfe,
            cfg=cfg,
            device=device,
            viz_dir=viz_dir,
            viz_every=args.viz_every,
            use_lpips=args.lpips,
            use_amp=use_amp,
        )

        sweep_results[nfe] = global_means

        # --- Stratification ---
        binned     = stratify_records(records)
        metric_keys = sorted(k for k in global_means)

        strat_means: Dict[str, Dict[str, float]] = {}
        for lbl in _BIN_LABELS:
            bin_records = binned[lbl]
            means = _bin_mean(bin_records, metric_keys)
            means["_count"] = len(bin_records)
            strat_means[lbl] = means

        # --- Summary table ---
        print_summary_table(
            global_means=global_means,
            strat_means=strat_means,
            metric_keys=metric_keys,
            nfe=nfe,
            total_samples=len(records),
        )

        # --- CSV ---
        save_csv(records, csv_dir / f"test_metrics_nfe{nfe}.csv")

        # --- Stratified CSV ---
        strat_csv_path = csv_dir / f"stratified_nfe{nfe}.csv"
        with open(strat_csv_path, "w", newline="") as f:
            fieldnames = ["coverage_bin", "count"] + metric_keys
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for lbl in _BIN_LABELS:
                bin_records = binned[lbl]
                row = {"coverage_bin": lbl, "count": len(bin_records)}
                row.update(_bin_mean(bin_records, metric_keys))
                writer.writerow(row)
            # Overall row
            overall = {"coverage_bin": "Overall", "count": len(records)}
            overall.update(global_means)
            writer.writerow(overall)
        log.info("Stratified CSV saved → %s", strat_csv_path)

    # =====================================================================
    # NFE sweep summary  (only meaningful if > 1 NFE evaluated)
    # =====================================================================
    if len(sweep_results) > 1:
        print_nfe_sweep_table(sweep_results, metric_keys=sorted(next(iter(sweep_results.values()))))

        # --- NFE sweep chart ---
        if _MPL:
            _plot_nfe_sweep(sweep_results, save_path=out_root / "nfe_sweep.png")

    log.info("Evaluation complete.  Results saved to %s", out_root)


def _plot_nfe_sweep(
    sweep: Dict[int, Dict[str, float]],
    save_path: Path,
) -> None:
    """Plot PSNR, SSIM, and LPIPS (if available) vs. NFE on a two-axis figure."""
    nfe_vals = sorted(sweep)
    psnr_vals = [sweep[n].get("psnr", float("nan")) for n in nfe_vals]
    ssim_vals = [sweep[n].get("ssim", float("nan")) for n in nfe_vals]
    sam_vals  = [sweep[n].get("sam",  float("nan")) for n in nfe_vals]
    has_lpips = any("lpips" in sweep[n] for n in nfe_vals)
    lpips_vals = [sweep[n].get("lpips", float("nan")) for n in nfe_vals] if has_lpips else None

    n_subplots = 3 + (1 if has_lpips else 0)
    fig, axes  = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 4))

    def _plot(ax, y, ylabel, color, marker="o"):
        ax.plot(nfe_vals, y, marker=marker, color=color, linewidth=2)
        ax.set_xlabel("NFE (inference steps)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(nfe_vals)
        ax.grid(True, alpha=0.3)

    _plot(axes[0], psnr_vals, "PSNR (dB) ↑",  "steelblue")
    _plot(axes[1], ssim_vals, "SSIM ↑",        "seagreen")
    _plot(axes[2], sam_vals,  "SAM (°) ↓",     "tomato")
    if has_lpips and lpips_vals is not None:
        _plot(axes[3], lpips_vals, "LPIPS ↓",  "mediumpurple")

    fig.suptitle("SAR-Guided Cloud Removal — NFE Tradeoff", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("NFE sweep chart saved → %s", save_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate the OT-ODE diffusion bridge on the test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", required=True, metavar="PATH",
        help="Path to training checkpoint (best.ckpt or epoch_NNNN.ckpt)",
    )
    parser.add_argument(
        "--nfe", nargs="+", type=int, default=None, metavar="N",
        help="Number of function evaluations for inference (default: from config)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run NFE sweep at 1, 3, 5 steps and print tradeoff table",
    )
    parser.add_argument(
        "--lpips", action="store_true",
        help="Compute LPIPS (requires `pip install lpips`; slow on CPU)",
    )
    parser.add_argument(
        "--viz-every", type=int, default=0, metavar="N",
        help="Save a comparison strip every N-th test sample (0 = disable)",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Output root for CSV and visualisations (overrides config)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help="Override batch size from config",
    )
    parser.add_argument(
        "--config", default=None, metavar="YAML",
        help="Extra YAML to merge on top of default.yaml (e.g. an ablation config)",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if not _OMEGACONF:
        raise SystemExit("omegaconf is required: pip install omegaconf")

    args, overrides = _parse_args()

    # --- Load config (same logic as train_bridge.py) ---
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    paths_cfg = config_path.parent / "paths.yaml"
    if paths_cfg.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(paths_cfg))
    if args.config:
        extra = Path(args.config)
        if not extra.exists():
            raise FileNotFoundError(f"--config file not found: {extra}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    evaluate(cfg, args)


if __name__ == "__main__":
    main()
