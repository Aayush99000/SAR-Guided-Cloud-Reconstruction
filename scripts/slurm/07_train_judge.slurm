#!/bin/bash
# =============================================================================
# 07_train_judge.slurm — Train the frozen land-cover judge on real clear images
# =============================================================================
# Needs the WorldCover labels from 06_fetch_worldcover.slurm.
#   sbatch scripts/slurm/07_train_judge.slurm
# Smoke test (50 steps, tiny val):
#   sbatch --export=ALL,MAX_STEPS=50 --time=00:30:00 scripts/slurm/07_train_judge.slurm
# =============================================================================
#SBATCH --job-name=train_judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/train_judge_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/train_judge_%j.err

set -euo pipefail

SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
WC_DIR="${SCRATCH}/Dataset/SEN12MSCRTS/worldcover"
MAX_STEPS="${MAX_STEPS:-0}"
if [[ "${MAX_STEPS}" -gt 0 ]]; then
    OUT="${SCRATCH}/outputs/checkpoints/judge_smoke/best.ckpt"
else
    OUT="${SCRATCH}/outputs/checkpoints/judge/best.ckpt"
fi
BACKUP_DIR="/home/katoch.aa/SAR-Diffusion-Bridge/checkpoints/judge"

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cloud-env
export TORCH_HOME="/scratch/katoch.aa/.cache/torch"
export HF_HOME="/scratch/katoch.aa/.cache/huggingface"
cd "${SCRATCH}"

echo "[INFO] Judge training | job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} | $(date)"
python -c "import torch; print(f'[INFO] PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)}')"

PYTHONPATH="${SCRATCH}" python train/train_judge.py \
    --config configs/sen12mscrts.yaml \
    --wc-dir "${WC_DIR}" \
    --out "${OUT}" \
    --workers 14 \
    --max-steps "${MAX_STEPS}"

if [[ "${MAX_STEPS}" -eq 0 ]]; then
    mkdir -p "${BACKUP_DIR}"
    cp "${OUT}" "${OUT%.ckpt}.json" "${BACKUP_DIR}/" \
        && echo "[INFO] Judge backed up to ${BACKUP_DIR}" \
        || echo "[WARN] Backup to home failed (quota?)"
fi
echo "[INFO] Done: $(date)"
