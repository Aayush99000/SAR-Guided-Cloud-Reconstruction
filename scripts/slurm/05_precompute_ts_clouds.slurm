#!/bin/bash
# =============================================================================
# 05_precompute_ts_clouds.slurm — s2cloudless masks + coverage for SEN12MS-CR-TS
# =============================================================================
# CPU only, ~316 core-hours total (~2.4 s per image on one core).
# Runs as a job array; tiles are sharded across tasks. Resumable: tiles whose
# outputs already exist are skipped. s2cloudless lives in its own venv
# (layered on cloud-env) so the training environment is never modified.
#
# Full run (4 tasks x 32 cores, ~2.5 h):
#   sbatch --array=0-3 scripts/slurm/05_precompute_ts_clouds.slurm
# Smoke test (1 tile, ~10 min):
#   sbatch --array=0 --export=ALL,LIMIT_TILES=1 scripts/slurm/05_precompute_ts_clouds.slurm
# =============================================================================
#SBATCH --job-name=ts_clouds
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/ts_clouds_%A_%a.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/ts_clouds_%A_%a.err

set -euo pipefail

SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
TS_ROOT="${SCRATCH}/Dataset/SEN12MSCRTS"
OUT_DIR="${TS_ROOT}/cloud_masks"
BACKUP_DIR="/home/katoch.aa/SAR-Diffusion-Bridge/data_index/SEN12MSCRTS_cloud_masks"
VENV="/scratch/katoch.aa/envs/s2cloudless"
LIMIT_TILES="${LIMIT_TILES:-0}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cloud-env

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[INFO] Creating s2cloudless venv at ${VENV}"
    python -m venv --system-site-packages "${VENV}"
    "${VENV}/bin/pip" install --quiet s2cloudless
fi
"${VENV}/bin/python" -c "import s2cloudless, rasterio; print('[INFO] s2cloudless OK')"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SEN12MS-CR-TS cloud-mask precompute — shard ${SHARD}/${NUM_SHARDS}"
echo "  Job ID : ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}"
echo "  Out    : ${OUT_DIR}"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

"${VENV}/bin/python" "${SCRATCH}/scripts/precompute_ts_clouds.py" \
    --root "${TS_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --shard "${SHARD}" --num-shards "${NUM_SHARDS}" \
    --limit-tiles "${LIMIT_TILES}"

# Scratch can be purged; keep a copy of the (small) index in home.
if [[ "${LIMIT_TILES}" == "0" ]]; then
    mkdir -p "${BACKUP_DIR}"
    rsync -a --exclude='*.tmp.*' "${OUT_DIR}/" "${BACKUP_DIR}/"
    echo "[INFO] Backed up to ${BACKUP_DIR}"
fi

echo "[INFO] Done: $(date)"
