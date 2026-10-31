#!/bin/bash
# =============================================================================
# 06_fetch_worldcover.slurm — ESA WorldCover labels for every SEN12MS-CR-TS patch
# =============================================================================
# CPU + network (windowed COG reads over HTTPS; nothing large is downloaded).
# Resumable: tiles whose label file exists are skipped.
#   sbatch scripts/slurm/06_fetch_worldcover.slurm
# Smoke test (first 2 tiles per split):
#   sbatch --export=ALL,LIMIT_TILES=2 scripts/slurm/06_fetch_worldcover.slurm
# =============================================================================
#SBATCH --job-name=fetch_wc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/fetch_wc_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/fetch_wc_%j.err

set -euo pipefail

SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
LIMIT_TILES="${LIMIT_TILES:-0}"

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cloud-env
cd "${SCRATCH}"

# Faster remote COG access: no directory listings, only .tif requests.
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif
export GDAL_HTTP_MAX_RETRY=5
export GDAL_HTTP_RETRY_DELAY=3

echo "[INFO] WorldCover fetch | job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} | $(date)"
PYTHONPATH="${SCRATCH}" python scripts/fetch_worldcover.py \
    --config configs/sen12mscrts.yaml \
    --workers 12 \
    --limit-tiles "${LIMIT_TILES}"
echo "[INFO] Labels: $(ls Dataset/SEN12MSCRTS/worldcover/train/*_wc.npy 2>/dev/null | wc -l) train / $(ls Dataset/SEN12MSCRTS/worldcover/test/*_wc.npy 2>/dev/null | wc -l) test tiles"
echo "[INFO] Done: $(date)"
