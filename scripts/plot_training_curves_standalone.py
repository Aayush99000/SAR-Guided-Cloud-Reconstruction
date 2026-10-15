"""
Standalone training curve plotter — data embedded from parsed SLURM logs.
Outputs: results/training_curves.png

Usage:
    python scripts/plot_training_curves_standalone.py
    python scripts/plot_training_curves_standalone.py --show
"""

import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Embedded data (parsed from SLURM logs, restart-deduplicated) ──────────── #
# Validation: best PSNR per epoch (filters restart artifacts where job
# resumed from an older checkpoint and re-validated the same epoch).
VAL = [
    # epoch, psnr,  ssim,   mae
    (5,   23.04, 0.2497, 0.0503),
    (8,   24.43, 0.2570, 0.0462),
    (10,  24.44, 0.2568, 0.0460),
    (12,  24.45, 0.2570, 0.0460),
    (14,  24.45, 0.2573, 0.0460),
    (16,  24.45, 0.2570, 0.0460),
    (18,  24.45, 0.2571, 0.0460),
    (20,  24.45, 0.2569, 0.0460),
    (22,  24.45, 0.2570, 0.0460),
    (24,  24.45, 0.2571, 0.0460),
    (26,  24.45, 0.2570, 0.0460),
    (28,  24.45, 0.2570, 0.0460),
    (30,  24.45, 0.2572, 0.0460),
    (32,  24.46, 0.2572, 0.0460),
    (34,  24.44, 0.2570, 0.0460),
    (36,  24.45, 0.2572, 0.0460),
    (38,  24.46, 0.2574, 0.0460),
    (40,  24.46, 0.2575, 0.0460),
    (42,  24.46, 0.2574, 0.0459),
    (44,  24.46, 0.2574, 0.0459),
    (46,  24.46, 0.2573, 0.0459),
    (48,  24.47, 0.2575, 0.0459),
    (50,  24.46, 0.2573, 0.0459),
    (52,  24.46, 0.2576, 0.0459),
    (54,  24.46, 0.2574, 0.0460),
    (56,  24.45, 0.2573, 0.0460),
    (58,  24.46, 0.2573, 0.0460),
    (60,  24.46, 0.2575, 0.0460),
    (62,  24.46, 0.2575, 0.0459),
    (64,  24.46, 0.2574, 0.0460),
    (66,  24.45, 0.2573, 0.0460),
    (68,  24.46, 0.2573, 0.0460),
    (70,  24.45, 0.2573, 0.0460),
    (75,  24.47, 0.2577, 0.0458),
    (80,  24.47, 0.2578, 0.0458),
    (85,  24.47, 0.2576, 0.0459),
    (90,  24.46, 0.2576, 0.0459),
    (95,  24.46, 0.2577, 0.0459),
    (100, 24.46, 0.2577, 0.0459),
]

# Training: first step-log per epoch (loss, l1, lr)
TRAIN = [
    # epoch, total_loss, l1_loss,  lr
    (1,   0.0667, 0.1506, 2.50e-6),
    (2,   0.0041, 0.0146, 5.00e-5),
    (3,   0.0038, 0.0126, 5.00e-5),
    (4,   0.0024, 0.0095, 4.99e-5),
    (5,   0.0020, 0.0064, 4.98e-5),
    (6,   0.0023, 0.0091, 4.97e-5),
    (7,   0.0014, 0.0035, 4.95e-5),
    (8,   0.0016, 0.0050, 4.93e-5),
    (9,   0.0018, 0.0030, 4.91e-5),
    (10,  0.0012, 0.0031, 4.89e-5),
    (11,  0.0010, 0.0031, 4.87e-5),
    (12,  0.0007, 0.0022, 4.84e-5),
    (13,  0.0010, 0.0025, 4.81e-5),
    (14,  0.0011, 0.0030, 4.78e-5),
    (15,  0.0006, 0.0020, 4.75e-5),
    (16,  0.0005, 0.0015, 4.71e-5),
    (17,  0.0007, 0.0016, 4.67e-5),
    (18,  0.0007, 0.0020, 4.63e-5),
    (19,  0.0005, 0.0014, 4.59e-5),
    (20,  0.0005, 0.0017, 4.55e-5),
    (21,  0.0005, 0.0021, 4.50e-5),
    (22,  0.0004, 0.0017, 4.45e-5),
    (23,  0.0004, 0.0016, 4.40e-5),
    (24,  0.0005, 0.0016, 4.35e-5),
    (25,  0.0005, 0.0015, 4.30e-5),
    (26,  0.0004, 0.0013, 4.24e-5),
    (27,  0.0003, 0.0015, 4.18e-5),
    (28,  0.0003, 0.0013, 4.12e-5),
    (29,  0.0003, 0.0009, 4.06e-5),
    (30,  0.0003, 0.0011, 4.00e-5),
    (31,  0.0004, 0.0014, 3.94e-5),
    (32,  0.0003, 0.0010, 3.87e-5),
    (33,  0.0003, 0.0011, 3.81e-5),
    (34,  0.0003, 0.0013, 3.74e-5),
    (35,  0.0002, 0.0007, 3.67e-5),
    (36,  0.0003, 0.0010, 3.60e-5),
    (37,  0.0007, 0.0014, 3.53e-5),
    (38,  0.0002, 0.0008, 3.46e-5),
    (39,  0.0001, 0.0006, 3.38e-5),
    (40,  0.0003, 0.0008, 3.31e-5),
    (41,  0.0003, 0.0010, 3.23e-5),
    (42,  0.0003, 0.0006, 3.16e-5),
    (43,  0.0002, 0.0007, 3.08e-5),
    (44,  0.0002, 0.0006, 3.01e-5),
    (45,  0.0002, 0.0006, 2.93e-5),
    (46,  0.0003, 0.0009, 2.85e-5),
    (47,  0.0001, 0.0006, 2.77e-5),
    (48,  0.0002, 0.0007, 2.69e-5),
    (49,  0.0001, 0.0005, 2.62e-5),
    (50,  0.0002, 0.0008, 2.54e-5),
    (51,  0.0002, 0.0005, 2.46e-5),
    (52,  0.0003, 0.0011, 2.38e-5),
    (53,  0.0001, 0.0005, 2.30e-5),
    (54,  0.0002, 0.0008, 2.22e-5),
    (55,  0.0002, 0.0007, 2.15e-5),
    (56,  0.0002, 0.0007, 2.07e-5),
    (57,  0.0001, 0.0004, 1.99e-5),
    (58,  0.0002, 0.0008, 1.91e-5),
    (59,  0.0003, 0.0006, 1.84e-5),
    (60,  0.0002, 0.0005, 1.76e-5),
    (61,  0.0002, 0.0006, 1.69e-5),
    (62,  0.0001, 0.0004, 1.61e-5),
    (63,  0.0002, 0.0006, 1.54e-5),
    (64,  0.0001, 0.0006, 1.47e-5),
    (65,  0.0001, 0.0004, 1.40e-5),
    (66,  0.0001, 0.0005, 1.33e-5),
    (67,  0.0001, 0.0005, 1.26e-5),
    (68,  0.0001, 0.0006, 1.19e-5),
    (69,  0.0001, 0.0004, 1.13e-5),
    (70,  0.0001, 0.0004, 1.06e-5),
    (71,  0.0001, 0.0005, 9.97e-6),
    (72,  0.0002, 0.0005, 9.35e-6),
    (73,  0.0001, 0.0003, 8.74e-6),
    (74,  0.0001, 0.0006, 8.15e-6),
    (75,  0.0001, 0.0003, 7.58e-6),
    (76,  0.0001, 0.0004, 7.02e-6),
    (77,  0.0001, 0.0004, 6.49e-6),
    (78,  0.0001, 0.0002, 5.97e-6),
    (79,  0.0001, 0.0002, 5.47e-6),
    (80,  0.0001, 0.0003, 4.98e-6),
    (81,  0.0001, 0.0004, 4.53e-6),
    (82,  0.0001, 0.0003, 4.09e-6),
    (83,  0.0001, 0.0002, 3.66e-6),
    (84,  0.0001, 0.0005, 3.27e-6),
    (85,  0.0000, 0.0002, 2.89e-6),
    (86,  0.0001, 0.0002, 2.53e-6),
    (87,  0.0001, 0.0003, 2.20e-6),
    (88,  0.0000, 0.0002, 1.89e-6),
    (89,  0.0002, 0.0004, 1.60e-6),
    (90,  0.0001, 0.0003, 1.33e-6),
    (91,  0.0001, 0.0003, 1.09e-6),
    (92,  0.0001, 0.0003, 8.74e-7),
    (93,  0.0001, 0.0002, 6.80e-7),
    (94,  0.0001, 0.0003, 5.10e-7),
    (95,  0.0001, 0.0002, 3.64e-7),
    (96,  0.0001, 0.0002, 2.43e-7),
    (97,  0.0001, 0.0002, 1.46e-7),
    (98,  0.0000, 0.0002, 7.33e-8),
    (99,  0.0001, 0.0002, 2.54e-8),
    (100, 0.0001, 0.0003, 2.33e-9),
]

# ── Plot ──────────────────────────────────────────────────────────────────── #
BLUE   = "#3b6bda"
TEAL   = "#0da882"
ORANGE = "#f07320"
PURPLE = "#7c5cbf"
RED    = "#e0435a"
AMBER  = "#f5a623"
FG     = "#1a2035"
FG2    = "#5a6380"
GRID   = "#e4e8f0"
BG     = "#f6f7fb"
SURF   = "#ffffff"


def _style(ax, ylabel=None, log_y=False):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=FG2, labelsize=8)
    ax.set_xlabel("Epoch", color=FG2, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG2, fontsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    if log_y:
        ax.set_yscale("log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 9,
        "axes.titleweight": "semibold",
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.dpi": 180,
    })

    ve   = [r[0] for r in VAL]
    vpsnr= [r[1] for r in VAL]
    vssim= [r[2] for r in VAL]
    vmae = [r[3] for r in VAL]

    te   = [r[0] for r in TRAIN]
    tloss= [r[1] for r in TRAIN]
    tl1  = [r[2] for r in TRAIN]
    tlr  = [r[3] for r in TRAIN]

    best_psnr = max(vpsnr)
    best_ep   = ve[vpsnr.index(best_psnr)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.44, wspace=0.30)

    # ── Val PSNR ─────────────────────────────────────────────────────────── #
    ax = axes[0, 0]
    _style(ax, ylabel="PSNR (dB)")
    ax.set_title("Validation PSNR")
    ax.plot(ve, vpsnr, color=BLUE, linewidth=1.8, zorder=3)
    ax.scatter([best_ep], [best_psnr], color=AMBER, s=60, zorder=5,
               edgecolors=SURF, linewidths=1.2)
    ax.annotate(
        f"Best {best_psnr:.2f} dB\n(epoch {best_ep})",
        xy=(best_ep, best_psnr),
        xytext=(best_ep + 9, best_psnr - 0.20),
        fontsize=7.5, color=FG2,
        arrowprops=dict(arrowstyle="-", color=FG2, lw=0.8),
    )
    ax.set_ylim(min(vpsnr) - 0.12, best_psnr + 0.18)
    ax.set_xlim(0, 105)

    # ── Val SSIM ─────────────────────────────────────────────────────────── #
    ax = axes[0, 1]
    _style(ax, ylabel="SSIM")
    ax.set_title("Validation SSIM")
    ax.plot(ve, vssim, color=TEAL, linewidth=1.8, zorder=3)
    ax.set_xlim(0, 105)

    # ── Train Loss ───────────────────────────────────────────────────────── #
    ax = axes[1, 0]
    _style(ax, ylabel="Loss", log_y=True)
    ax.set_title("Training Loss  (log scale)")
    ax.plot(te, tloss, color=ORANGE, linewidth=1.6, label="Total", zorder=3)
    ax.plot(te, tl1,   color=PURPLE, linewidth=1.6, label="L1",
            linestyle="--", zorder=3)
    ax.legend(fontsize=8, frameon=False, labelcolor=FG2)
    ax.set_xlim(0, 105)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    # ── LR Schedule ──────────────────────────────────────────────────────── #
    ax = axes[1, 1]
    _style(ax, ylabel="Learning Rate", log_y=True)
    ax.set_title("LR Schedule  (log scale)")
    ax.plot(te, tlr, color=RED, linewidth=1.6, zorder=3)
    ax.fill_between(te, tlr, alpha=0.12, color=RED)
    ax.set_xlim(0, 105)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    fig.suptitle(
        "v3 + LPIPS · OT-ODE Diffusion Bridge · SEN12MS-CR · 100 epochs · V100 32 GB",
        fontsize=9.5, color=FG, y=1.01, fontweight="semibold",
    )

    out = Path(__file__).parent.parent / "results" / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
