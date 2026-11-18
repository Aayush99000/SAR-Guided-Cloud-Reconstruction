#!/bin/bash
# =============================================================================
# 09_pack_ts.slurm — Pack SEN12MS-CR-TS tiles into per-tile .npy arrays
# =============================================================================
# One-off CPU job; see scripts/pack_ts.py.  Training reads packed tiles from
# paths.ts_pack_dir and the rest from the GeoTIFFs, so it is safe to run while
# training, and to resubmit: finished tiles are skipped.
#   sbatch scripts/slurm/09_pack_ts.slurm          # train/ (official train + val tiles)
#   sbatch scripts/slurm/09_pack_ts.slurm test     # test/
# =============================================================================
#SBATCH --job-name=pack_ts
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/pack_ts_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/pack_ts_%j.err

set -euo pipefail

SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
SPLIT="${1:-train}"

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cloud-env

cd "${SCRATCH}"
python scripts/pack_ts.py --config configs/default.yaml configs/sen12mscrts.yaml \
    --out "${SCRATCH}/Dataset/SEN12MSCRTS/packed" --split "${SPLIT}" --workers "${SLURM_CPUS_PER_TASK}"
echo "[INFO] Packed split ${SPLIT}: $(du -sh "${SCRATCH}/Dataset/SEN12MSCRTS/packed/${SPLIT}" | cut -f1)"
