"""
Plot training and validation metric curves from SLURM log files.

Usage:
    python scripts/plot_training_curves.py --log-dir logs/slurm
    python scripts/plot_training_curves.py --log-dir logs/slurm --out results/curves.png
    python scripts/plot_training_curves.py --log-dir logs/slurm --show

The script parses all train_*.err files, deduplicates restart artifacts
(keeps the best PSNR per epoch), and produces a 4-panel figure:
  - Validation PSNR
  - Validation SSIM
  - Training loss  (total + L1, log scale)
  - Learning-rate schedule (log scale)
"""

import argparse
import glob
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
_VAL_RE  = re.compile(
    r"\[Epoch\s+(\d+)\]\s+val\s+PSNR=([\d.]+)\s+dB\s+SSIM=([\d.]+)\s+MAE=([\d.]+)"
)
_STEP_RE = re.compile(
    r"\[E(\d+)\s+step\s+\d+\]\s+loss=([\d.]+)\s+l1=([\d.]+)\s+lr=([\d.e+\-]+)"
)


def parse_logs(log_dir: str):
    """Return (val_records, train_records) from all train_*.err in log_dir."""
    files = sorted(glob.glob(str(Path(log_dir) / "train_*.err")))
    if not files:
        sys.exit(f"[ERROR] No train_*.err files found in: {log_dir}")

    # val: keep best PSNR per epoch (filters restart duplicates)
    val_best: dict[int, dict] = {}
    # train: keep first step-log per epoch (representative sample)
    train_first: dict[int, dict] = {}

    for fpath in files:
        try:
            text = Path(fpath).read_text(errors="replace")
        except OSError:
            continue

        for m in _VAL_RE.finditer(text):
            epoch = int(m.group(1))
            psnr  = float(m.group(2))
            ssim  = float(m.group(3))
            mae   = float(m.group(4))
            # skip corrupt zero entries
            if psnr < 1.0:
                continue
            if epoch not in val_best or psnr > val_best[epoch]["psnr"]:
                val_best[epoch] = dict(epoch=epoch, psnr=psnr, ssim=ssim, mae=mae)

        for m in _STEP_RE.finditer(text):
            epoch = int(m.group(1))
            if epoch not in train_first:
                train_first[epoch] = dict(
                    epoch=epoch,
                    loss=float(m.group(2)),
                    l1=float(m.group(3)),
                    lr=float(m.group(4)),
                )

    val_records   = sorted(val_best.values(),   key=lambda x: x["epoch"])
    train_records = sorted(train_first.values(), key=lambda x: x["epoch"])

    if not val_records:
        sys.exit("[ERROR] No validation log lines found — check --log-dir path.")
    if not train_records:
        sys.exit("[ERROR] No training step log lines found — check --log-dir path.")

    return val_records, train_records


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #
BLUE   = "#4a7fd4"
TEAL   = "#29b388"
ORANGE = "#e87722"
PURPLE = "#8b5bbf"
AMBER  = "#f5a623"

FG     = "#1a2035"
FG2    = "#5a6380"
GRID   = "#e4e8f0"
BG     = "#f6f7fb"
SURF   = "#ffffff"


def _style_ax(ax, ylabel=None, log_y=False):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=FG2, labelsize=8)
    ax.set_xlabel("Epoch", color=FG2, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG2, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))


def plot_curves(val_records, train_records, out_path=None, show=False):
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size":   9,
        "axes.titlesize": 9,
        "axes.titleweight": "semibold",
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.dpi": 150,
    })

    val_e    = [r["epoch"] for r in val_records]
    val_psnr = [r["psnr"]  for r in val_records]
    val_ssim = [r["ssim"]  for r in val_records]

    tr_e    = [r["epoch"] for r in train_records]
    tr_loss = [r["loss"]  for r in train_records]
    tr_l1   = [r["l1"]    for r in train_records]
    tr_lr   = [r["lr"]    for r in train_records]

    best_psnr = max(val_psnr)
    best_ep   = val_e[val_psnr.index(best_psnr)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    fig.subplots_adjust(hspace=0.42, wspace=0.30)

    # ── PSNR ──────────────────────────────────────────────────────────────── #
    ax = axes[0, 0]
    _style_ax(ax, ylabel="PSNR (dB)")
    ax.set_title("Validation PSNR")
    ax.plot(val_e, val_psnr, color=BLUE, linewidth=1.8, zorder=3)
    ax.scatter([best_ep], [best_psnr], color=AMBER, s=55, zorder=5,
               edgecolors=SURF, linewidths=1.2)
    ax.annotate(
        f"Best {best_psnr:.2f} dB\n(epoch {best_ep})",
        xy=(best_ep, best_psnr),
        xytext=(best_ep + 8, best_psnr - 0.18),
        fontsize=7, color=FG2,
        arrowprops=dict(arrowstyle="-", color=FG2, lw=0.8),
    )
    y_lo = min(val_psnr) - 0.1
    y_hi = best_psnr + 0.15
    ax.set_ylim(y_lo, y_hi)

    # ── SSIM ──────────────────────────────────────────────────────────────── #
    ax = axes[0, 1]
    _style_ax(ax, ylabel="SSIM")
    ax.set_title("Validation SSIM")
    ax.plot(val_e, val_ssim, color=TEAL, linewidth=1.8, zorder=3)

    # ── Loss ──────────────────────────────────────────────────────────────── #
    ax = axes[1, 0]
    _style_ax(ax, ylabel="Loss", log_y=True)
    ax.set_title("Training Loss  (log scale)")
    ax.plot(tr_e, tr_loss, color=ORANGE, linewidth=1.5, label="Total", zorder=3)
    ax.plot(tr_e, tr_l1,   color=PURPLE, linewidth=1.5, label="L1",    zorder=3,
            linestyle="--")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=FG2)

    # ── LR ────────────────────────────────────────────────────────────────── #
    ax = axes[1, 1]
    _style_ax(ax, ylabel="Learning Rate", log_y=True)
    ax.set_title("LR Schedule  (log scale)")
    ax.plot(tr_e, tr_lr, color="#e04060", linewidth=1.5, zorder=3)
    ax.fill_between(tr_e, tr_lr, alpha=0.12, color="#e04060")

    # ── Figure title ──────────────────────────────────────────────────────── #
    fig.suptitle(
        "v3 + LPIPS — OT-ODE Diffusion Bridge  ·  SEN12MS-CR  ·  100 epochs",
        fontsize=10, color=FG, y=0.99, fontweight="semibold",
    )

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[OK] Saved → {out_path}")

    if show:
        plt.show()
    elif not out_path:
        # default: save next to script
        default = Path(__file__).parent.parent / "results" / "training_curves.png"
        default.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(default, bbox_inches="tight")
        print(f"[OK] Saved → {default}")

    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", default="logs/slurm",
                        help="Directory containing train_*.err files (default: logs/slurm)")
    parser.add_argument("--out",  default=None,
                        help="Output path for the PNG (default: results/training_curves.png)")
    parser.add_argument("--show", action="store_true",
                        help="Display the figure interactively instead of (or as well as) saving")
    args = parser.parse_args()

    val_records, train_records = parse_logs(args.log_dir)

    print(f"[INFO] Val checkpoints : {len(val_records)} epochs")
    print(f"[INFO] Train checkpoints: {len(train_records)} epochs")
    best = max(val_records, key=lambda r: r["psnr"])
    print(f"[INFO] Best val PSNR   : {best['psnr']:.2f} dB  (epoch {best['epoch']})")

    plot_curves(val_records, train_records, out_path=args.out, show=args.show)


if __name__ == "__main__":
    main()
