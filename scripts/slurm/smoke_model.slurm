#!/bin/bash
# Smoke test for the model variants (see scripts/smoke_model.py):
#   sbatch scripts/slurm/smoke_model.slurm
#SBATCH --job-name=smoke_model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/smoke_model_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/smoke_model_%j.err

set -euo pipefail
SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cloud-env
export TORCH_HOME="/scratch/katoch.aa/.cache/torch"
cd "${SCRATCH}"

PYTHONPATH="${SCRATCH}" python -m models.backbone.unet
for overlay in mask temporal_concat temporal; do
    echo "════════ ${overlay}"
    V1_ARG=()
    [[ "${overlay}" == mask ]] && V1_ARG=(--v1-ckpt "${SCRATCH}/outputs/checkpoints/bridge_ts_v1/best.ckpt")
    PYTHONPATH="${SCRATCH}" python scripts/smoke_model.py "${V1_ARG[@]}" \
        --config configs/sen12mscrts.yaml "configs/sen12mscrts_${overlay}.yaml"
done
