#!/usr/bin/env bash
# =============================================================================
# 02_train_bridge.slurm — Chain-trained diffusion bridge (self-resubmitting)
# =============================================================================
# Because the cluster enforces an 8-hour wall time, this script trains for one
# session and automatically resubmits itself for the next session until all
# epochs are done.  Each session saves last.ckpt; the next session resumes
# from it.  No code changes to train_bridge.py are needed.
#
# Chain structure (example: 100 epochs, ~16 epochs/session):
#
#   Job A (epochs 1-16)  →  Job B (epochs 17-32)  →  …  →  Job G (epochs 97-100)
#    saves last.ckpt          resumes, saves              done, saves best.ckpt
#
# The script detects completion by reading the epoch count from last.ckpt and
# comparing it to TOTAL_EPOCHS.  It does NOT resubmit once training is done.
#
# ── First submission ──────────────────────────────────────────────────────────
#   sbatch scripts/slurm/02_train_bridge.slurm
#
# ── After download job ────────────────────────────────────────────────────────
#   sbatch --dependency=afterok:<download_jid> scripts/slurm/02_train_bridge.slurm
#
# ── Manual resume (e.g. after cluster outage) ─────────────────────────────────
#   sbatch --export=ALL,RESUME=outputs/checkpoints/bridge/last.ckpt \
#          scripts/slurm/02_train_bridge.slurm
#
# ── Dry-run (print resubmit command without running) ─────────────────────────
#   DRY_RUN=1 bash scripts/slurm/02_train_bridge.slurm
# =============================================================================
#SBATCH --job-name=sar_bridge_train
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1         # V100 SXM2 nodes idle on Discovery; L40 nodes often busy
#SBATCH --time=01:00:00                # sharing partition max = 1 h
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/slurm

DRY_RUN="${DRY_RUN:-0}"

# =============================================================================
# CONFIGURATION — edit these to match your setup
# =============================================================================
TOTAL_EPOCHS=100                        # must match training.num_epochs in config
SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
CKPT_DIR="${SCRATCH}/outputs/checkpoints/bridge"
LAST_CKPT="${CKPT_DIR}/latest.ckpt"    # train_bridge.py writes latest.ckpt every epoch
THIS_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"

# GPU/cluster tuning
NUM_WORKERS=8
DATA_ROOT_OVERRIDE="paths.output_root=${SCRATCH}/outputs"
# =============================================================================

# --- Environment -------------------------------------------------------------
module purge
module load anaconda3/2022.05           # check available: module avail anaconda3
# Initialise conda in this non-interactive shell (required for conda activate)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cloud-env

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SAR Bridge Training Session"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Node   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPUs   : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -c "
import torch
d = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print(f'[INFO] PyTorch {torch.__version__} | {d}')
"

# --- Determine resume path ---------------------------------------------------
# Priority: env var RESUME → last.ckpt on disk → fresh start
if [[ -n "${RESUME:-}" && -f "${RESUME}" ]]; then
    CKPT_ARG="--resume ${RESUME}"
    echo "[INFO] Resuming from env: ${RESUME}"
elif [[ -f "${LAST_CKPT}" ]]; then
    CKPT_ARG="--resume ${LAST_CKPT}"
    echo "[INFO] Resuming from disk: ${LAST_CKPT}"
else
    CKPT_ARG=""
    echo "[INFO] Fresh training run (no checkpoint found)"
fi

# --- Train for this session --------------------------------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] would run: python train/train_bridge.py --config configs/default.yaml ${CKPT_ARG} ..."
else
    python train/train_bridge.py \
        --config configs/default.yaml \
        training.num_workers="${NUM_WORKERS}" \
        logging.use_wandb=true \
        ${DATA_ROOT_OVERRIDE:-} \
        ${CKPT_ARG}
fi

echo "[INFO] Session end: $(date)"

# --- Check if training is complete -------------------------------------------
# Read the epoch number saved in last.ckpt using Python.
COMPLETED_EPOCHS=0
if [[ -f "${LAST_CKPT}" && "${DRY_RUN}" != "1" ]]; then
    COMPLETED_EPOCHS=$(python -c "
import torch, sys
try:
    ckpt = torch.load('${LAST_CKPT}', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except Exception as e:
    print(0)
")
fi

echo "[INFO] Completed epochs: ${COMPLETED_EPOCHS} / ${TOTAL_EPOCHS}"

if [[ "${COMPLETED_EPOCHS}" -ge "${TOTAL_EPOCHS}" ]]; then
    echo "[INFO] Training complete — no resubmission needed."
    echo "[INFO] Best checkpoint: ${CKPT_DIR}/best.ckpt"

    # --- Auto-trigger eval and ablations (if scripts exist) ---------------
    SLURM_DIR="$(dirname "${THIS_SCRIPT}")"
    if [[ "${DRY_RUN}" != "1" ]]; then
        if [[ -f "${SLURM_DIR}/03_evaluate.slurm" ]]; then
            EVAL_JOB=$(sbatch "${SLURM_DIR}/03_evaluate.slurm")
            echo "[INFO] Submitted eval:      ${EVAL_JOB}"
        fi
        if [[ -f "${SLURM_DIR}/04_ablations.slurm" ]]; then
            ABL_JOB=$(sbatch "${SLURM_DIR}/04_ablations.slurm")
            echo "[INFO] Submitted ablations: ${ABL_JOB}  (array 0-4)"
        fi
    else
        echo "[dry-run] would submit 03_evaluate.slurm and 04_ablations.slurm"
    fi
    exit 0
fi

# --- Resubmit for the next session -------------------------------------------
NEXT_CMD="sbatch --export=ALL,RESUME=${LAST_CKPT} ${THIS_SCRIPT}"   # LAST_CKPT=latest.ckpt
echo ""
echo "[INFO] Epochs remaining: $(( TOTAL_EPOCHS - COMPLETED_EPOCHS ))"
echo "[INFO] Resubmitting:  ${NEXT_CMD}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] would resubmit"
else
    NEXT_JOB=$(${NEXT_CMD})
    echo "[INFO] Next job submitted: ${NEXT_JOB}"
fi
