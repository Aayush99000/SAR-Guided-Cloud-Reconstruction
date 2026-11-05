#!/bin/bash
# Smoke test for model.mask_input (see scripts/smoke_mask_input.py):
#   sbatch scripts/slurm/smoke_mask_input.slurm
#SBATCH --job-name=smoke_mask
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/smoke_mask_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/smoke_mask_%j.err

set -euo pipefail
SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cloud-env
export TORCH_HOME="/scratch/katoch.aa/.cache/torch"
cd "${SCRATCH}"

PYTHONPATH="${SCRATCH}" python -m models.backbone.unet
PYTHONPATH="${SCRATCH}" python scripts/smoke_mask_input.py \
    --v1-ckpt "${SCRATCH}/outputs/checkpoints/bridge_ts_v1/best.ckpt"
