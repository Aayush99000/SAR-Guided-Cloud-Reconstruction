#!/usr/bin/env bash
# =============================================================================
# 03_evaluate.slurm — Evaluate best checkpoint on the test split
# =============================================================================
# Runs evaluate.py at NFE={1,3,5,10}, computes PSNR/SSIM/MAE/SAM/LPIPS,
# produces stratified metrics (5 cloud-coverage bins), saves visualizations,
# and writes per-sample CSV.
#
# Expected runtime: ~1-3 h on a single GPU (test set size dependent).
#
# Submit:
#   sbatch scripts/slurm/03_evaluate.slurm
#
# Run as dependency (after training):
#   sbatch --dependency=afterok:<train_job_id> scripts/slurm/03_evaluate.slurm
#
# Override checkpoint path:
#   sbatch --export=CKPT=outputs/ablations/backbone/nafblock_vim/best.ckpt \
#          scripts/slurm/03_evaluate.slurm
# =============================================================================
#SBATCH --job-name=sar_bridge_eval
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR}"
cd "${ROOT_DIR}"

mkdir -p logs/slurm

# --- Environment ---------------------------------------------------------
module purge
module load anaconda3/2022.05           # check available: module avail anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cloud-env

echo "[INFO] Job ID:  ${SLURM_JOB_ID}"
echo "[INFO] Node:    ${SLURMD_NODENAME}"
echo "[INFO] Start:   $(date)"

# --- Checkpoint ----------------------------------------------------------
CKPT="${CKPT:-outputs/checkpoints/bridge/best.ckpt}"
OUT_DIR="${OUT_DIR:-outputs/eval}"

if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    echo "        Run 02_train_bridge.slurm first, or set CKPT=<path>"
    exit 1
fi

echo "[INFO] Checkpoint: ${CKPT}"
echo "[INFO] Output dir: ${OUT_DIR}"

# --- Evaluate at NFE=1,3,5,10 with LPIPS ---------------------------------
python train/evaluate.py \
    --config configs/default.yaml \
    --ckpt  "${CKPT}" \
    --nfe   1 3 5 10 \
    --lpips \
    --viz-every 50 \
    --out-dir "${OUT_DIR}"

echo "[INFO] Done:    $(date)"
echo "[INFO] Results written to ${OUT_DIR}/"
