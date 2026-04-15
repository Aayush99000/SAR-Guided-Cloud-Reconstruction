#!/usr/bin/env bash
# =============================================================================
# 01_download_data.slurm — Download SEN12MS-CR dataset from TUM server
# =============================================================================
# CPU-only job; no GPU needed.  Runs download_data.sh, which has a 3-strategy
# fallback (curl WebDAV → wget → manual prompt) and skips already-downloaded
# archives.
#
# Typical wall time: 2-6 h depending on TUM server load (~300 GB total).
#
# Submit:
#   sbatch scripts/slurm/01_download_data.slurm
# =============================================================================
#SBATCH --job-name=sen12ms_download
#SBATCH --partition=short               # CPU-only; adjust to your cluster's name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/download_%j.out
#SBATCH --error=logs/slurm/download_%j.err

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/slurm

# --- Environment ---------------------------------------------------------
module purge
# Uncomment the line that matches your cluster's module system:
# module load anaconda3/2023.03          # Northeastern Discovery
# module load miniconda/23.1.0           # generic
conda activate cloud-env               # adjust to your env name

# --- Optional: redirect data root via override ---------------------------
# By default uses paths.yaml → /data/SEN12MS-CR
# Override at submit time: sbatch --export=DATA_ROOT=/scratch/$USER/SEN12MS-CR ...
DATA_ROOT="${DATA_ROOT:-/data/SEN12MS-CR}"

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node:   ${SLURMD_NODENAME}"
echo "[INFO] Target: ${DATA_ROOT}"
echo "[INFO] Start:  $(date)"

bash scripts/download_data.sh --out-dir "${DATA_ROOT}"

echo "[INFO] Done:   $(date)"
