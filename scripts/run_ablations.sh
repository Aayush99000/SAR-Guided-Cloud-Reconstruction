#!/usr/bin/env bash
# =============================================================================
# run_ablations.sh — Train and evaluate all ablation study variants
# =============================================================================
#
# Runs five ablation studies sequentially on a single GPU.  Each variant is
# a separate training run with its own checkpoint directory.  On a multi-GPU
# cluster, launch the five studies in parallel (one per GPU) by passing
# --study <name> to this script.
#
# Prerequisites
# -------------
#   conda activate cloud-env      # or your venv
#   python -m pip install omegaconf wandb tqdm lpips   # if not installed
#
# Usage
# -----
#   # Run ALL ablations (sequential, ~150 GPU-hours on A100):
#   bash scripts/run_ablations.sh
#
#   # Run a single study:
#   bash scripts/run_ablations.sh --study backbone
#   bash scripts/run_ablations.sh --study fusion
#   bash scripts/run_ablations.sh --study loss
#   bash scripts/run_ablations.sh --study nfe
#   bash scripts/run_ablations.sh --study schedule
#
#   # Dry-run (print commands without executing):
#   DRY_RUN=1 bash scripts/run_ablations.sh
#
# Output layout
# -------------
#   outputs/ablations/
#     backbone/
#       nafblock_vim/        ← variant (a)
#       nafblock_attn/       ← variant (b)
#       nafblock_none/       ← variant (c)
#     fusion/
#       sfblock/             ← variant (a)
#       early_concat/        ← variant (b)
#       no_sar/              ← variant (c)
#     loss/
#       cloud_aware_full/    ← variant (a)
#       uniform_mse/         ← variant (b)
#       cloud_aware_mse/     ← variant (c)
#     schedule/
#       cosine/              ← variant (a)
#       sine/                ← variant (b)
#       linear/              ← variant (c)
#     nfe/                   ← evaluation outputs (no training)
#       nfe_sweep.png
#       metrics/
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
TRAIN="${PYTHON} train/train_bridge.py"
EVAL="${PYTHON} train/evaluate.py"
ABLATION_DIR="outputs/ablations"

# Activate conda/venv if present
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "[INFO] Using conda env: ${CONDA_DEFAULT_ENV}"
fi

# Parse --study flag
STUDY="${1:-all}"
if [[ "${1:-}" == "--study" ]]; then
    STUDY="${2:-all}"
fi

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo; }
run()  {
    echo "  $ $*"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  [dry-run — skipped]"
    else
        "$@"
    fi
}
ckpt() { echo "${ABLATION_DIR}/${1}/best.ckpt"; }


# ---------------------------------------------------------------------------
# Study 1: Backbone Architecture
# ---------------------------------------------------------------------------
run_backbone() {
    local CFG="configs/ablations/ablation_backbone.yaml"
    log "Ablation 1/5 — Backbone Architecture"

    # (a) NAFBlock + VimSSM  [our full model]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/backbone/nafblock_vim" \
        logging.run_name="backbone_a_nafblock_vim"

    # (b) NAFBlock + Self-Attention  [DB-CR-style bottleneck]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/backbone/nafblock_attn" \
        logging.run_name="backbone_b_nafblock_attn" \
        model.bottleneck_type=attention

    # (c) NAFBlock only  [no global-context bottleneck]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/backbone/nafblock_none" \
        logging.run_name="backbone_c_nafblock_none" \
        model.bottleneck_type=none \
        model.num_vim_blocks=0

    # --- Evaluate all three ---
    for variant in nafblock_vim nafblock_attn nafblock_none; do
        ckpt_path="${ABLATION_DIR}/backbone/${variant}/best.ckpt"
        run ${EVAL} \
            --config "${CFG}" \
            --ckpt "${ckpt_path}" \
            --nfe 5 \
            --out-dir "${ABLATION_DIR}/backbone/${variant}/eval"
    done
}


# ---------------------------------------------------------------------------
# Study 2: SAR Fusion Strategy
# ---------------------------------------------------------------------------
run_fusion() {
    local CFG="configs/ablations/ablation_fusion.yaml"
    log "Ablation 2/5 — SAR Fusion Strategy"

    # (a) SFBlock cross-attention  [our dual-branch design]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/fusion/sfblock" \
        logging.run_name="fusion_a_sfblock"

    # (b) Early concatenation only  [SAR in input but no deep fusion]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/fusion/early_concat" \
        logging.run_name="fusion_b_early_concat" \
        model.fusion_mode=early_concat

    # (c) No SAR conditioning  [optical-only baseline]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/fusion/no_sar" \
        logging.run_name="fusion_c_no_sar" \
        model.fusion_mode=none

    # --- Evaluate all three ---
    for variant in sfblock early_concat no_sar; do
        ckpt_path="${ABLATION_DIR}/fusion/${variant}/best.ckpt"
        run ${EVAL} \
            --config "${CFG}" \
            --ckpt "${ckpt_path}" \
            --nfe 5 \
            --out-dir "${ABLATION_DIR}/fusion/${variant}/eval"
    done
}


# ---------------------------------------------------------------------------
# Study 3: Loss Function
# ---------------------------------------------------------------------------
run_loss() {
    local CFG="configs/ablations/ablation_loss.yaml"
    log "Ablation 3/5 — Loss Function"

    # (a) Cloud-aware weighted MSE + SSIM  [our full loss]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/loss/cloud_aware_full" \
        logging.run_name="loss_a_cloud_aware_full"
    # loss.alpha=0.8, lambda_mse=0.5, lambda_ssim=0.5  ← from ablation_loss.yaml

    # (b) Uniform MSE  [W=0.5 everywhere, no SSIM]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/loss/uniform_mse" \
        logging.run_name="loss_b_uniform_mse" \
        loss.alpha=0.5 \
        loss.lambda_mse=1.0 \
        loss.lambda_ssim=0.0

    # (c) Cloud-aware weighting, MSE only  [no SSIM term]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/loss/cloud_aware_mse" \
        logging.run_name="loss_c_cloud_aware_mse" \
        loss.alpha=0.8 \
        loss.lambda_mse=1.0 \
        loss.lambda_ssim=0.0

    # --- Evaluate all three ---
    for variant in cloud_aware_full uniform_mse cloud_aware_mse; do
        ckpt_path="${ABLATION_DIR}/loss/${variant}/best.ckpt"
        run ${EVAL} \
            --config "${CFG}" \
            --ckpt "${ckpt_path}" \
            --nfe 5 \
            --out-dir "${ABLATION_DIR}/loss/${variant}/eval"
    done
}


# ---------------------------------------------------------------------------
# Study 4: NFE Sweep  (evaluation only — reuses the default trained model)
# ---------------------------------------------------------------------------
run_nfe() {
    local CFG="configs/ablations/ablation_nfe.yaml"
    log "Ablation 4/5 — NFE Sweep (evaluation only)"

    # Use the best checkpoint from the main training run
    BEST_CKPT="${BEST_CKPT:-outputs/checkpoints/bridge/best.ckpt}"

    if [[ ! -f "${BEST_CKPT}" ]]; then
        echo "[WARN] Checkpoint not found at ${BEST_CKPT}"
        echo "       Train the full model first:  bash scripts/run_training.sh"
        echo "       Or set BEST_CKPT=path/to/best.ckpt"
        return 1
    fi

    run ${EVAL} \
        --config "${CFG}" \
        --ckpt "${BEST_CKPT}" \
        --nfe 1 3 5 10 \
        --lpips \
        --viz-every 50 \
        --out-dir "${ABLATION_DIR}/nfe"
}


# ---------------------------------------------------------------------------
# Study 5: Alpha Schedule
# ---------------------------------------------------------------------------
run_schedule() {
    local CFG="configs/ablations/ablation_schedule.yaml"
    log "Ablation 5/5 — Alpha Schedule"

    # (a) Cosine  [DB-CR default / ours]
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/schedule/cosine" \
        logging.run_name="schedule_a_cosine"
    # diffusion.alpha_schedule_type=cosine  ← from ablation_schedule.yaml

    # (b) Sine
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/schedule/sine" \
        logging.run_name="schedule_b_sine" \
        diffusion.alpha_schedule_type=sine

    # (c) Linear
    run ${TRAIN} \
        --config "${CFG}" \
        paths.bridge_ckpt_dir="${ABLATION_DIR}/schedule/linear" \
        logging.run_name="schedule_c_linear" \
        diffusion.alpha_schedule_type=linear

    # --- Evaluate all three at 1, 5, 10 NFE to show schedule × NFE interaction ---
    for variant in cosine sine linear; do
        ckpt_path="${ABLATION_DIR}/schedule/${variant}/best.ckpt"
        run ${EVAL} \
            --config "${CFG}" \
            --ckpt "${ckpt_path}" \
            --nfe 1 5 10 \
            --out-dir "${ABLATION_DIR}/schedule/${variant}/eval"
    done
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${STUDY}" in
    all)
        run_backbone
        run_fusion
        run_loss
        run_nfe
        run_schedule
        ;;
    backbone)  run_backbone  ;;
    fusion)    run_fusion    ;;
    loss)      run_loss      ;;
    nfe)       run_nfe       ;;
    schedule)  run_schedule  ;;
    *)
        echo "Unknown study: ${STUDY}"
        echo "Usage: bash scripts/run_ablations.sh [--study backbone|fusion|loss|nfe|schedule|all]"
        exit 1
        ;;
esac

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All ablations complete.  Results in ${ABLATION_DIR}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
