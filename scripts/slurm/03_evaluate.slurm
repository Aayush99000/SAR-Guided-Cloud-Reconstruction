#!/usr/bin/env bash
# =============================================================================
# 03_evaluate.slurm — Evaluate best checkpoint on the test split (Explorer)
# =============================================================================
# Runs evaluate.py at NFE={1,3,5,10}, computes PSNR/SSIM/MAE/SAM/LPIPS,
# produces stratified metrics (5 cloud-coverage bins), saves visualizations,
# and writes per-sample CSV.
#
# Submit:
#   sbatch scripts/slurm/03_evaluate.slurm
#
# Override checkpoint path:
#   sbatch --export=CKPT=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/outputs/checkpoints/bridge/best.ckpt \
#          scripts/slurm/03_evaluate.slurm
# =============================================================================
#SBATCH --job-name=sar_bridge_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR}"
cd "${ROOT_DIR}"
mkdir -p logs/slurm

SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
CKPT="${CKPT:-${SCRATCH}/outputs/checkpoints/bridge/best.ckpt}"
OUT_DIR="${OUT_DIR:-${SCRATCH}/outputs/eval}"
DATA_ROOT_OVERRIDE="paths.output_root=${SCRATCH}/outputs"

module purge
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cloud-env

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SAR Bridge Evaluation — Explorer H200"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Node      : ${SLURMD_NODENAME}"
echo "  Checkpoint: ${CKPT}"
echo "  Output    : ${OUT_DIR}"
echo "  Start     : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -c "import torch; print(f'[INFO] PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)}')"

if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    exit 1
fi

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" python train/evaluate.py \
    --config configs/default.yaml \
    --ckpt   "${CKPT}" \
    --nfe    1 3 5 10 \
    --viz-every 50 \
    --out-dir "${OUT_DIR}" \
    training.num_workers=8 \
    data.multitemporal_sar=false \
    data.sar_bands=2 \
    "${DATA_ROOT_OVERRIDE}"

echo "[INFO] Done: $(date)"
echo "[INFO] Results written to ${OUT_DIR}/"
