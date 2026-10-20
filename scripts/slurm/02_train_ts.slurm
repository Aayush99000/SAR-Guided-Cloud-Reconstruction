#!/usr/bin/env bash
# =============================================================================
# 02_train_ts.slurm — Train the OT-ODE bridge on SEN12MS-CR-TS (3 dates, 6ch SAR)
# =============================================================================
# Self-chaining 8 h sessions; resumes from latest.ckpt. Submit from the repo root:
#   cd /scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction
#   sbatch scripts/slurm/02_train_ts.slurm
# Requires the cloud-mask index from 05_precompute_ts_clouds.slurm.
# =============================================================================
#SBATCH --job-name=bridge_ts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/train_ts_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/train_ts_%j.err

# Clean-environment relaunch: prevents "Argument list too long" across long job chains.
if [[ -z "${_CLEAN_ENV:-}" ]]; then
    exec env -i \
        HOME="${HOME}" \
        USER="${USER}" \
        LOGNAME="${LOGNAME:-${USER}}" \
        SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
        SLURMD_NODENAME="${SLURMD_NODENAME:-}" \
        SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}" \
        _CLEAN_ENV=1 \
        /bin/bash "$0" "$@"
fi

set -euo pipefail
SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
ROOT_DIR="${SCRATCH}"
cd "${ROOT_DIR}"
mkdir -p logs/slurm

TOTAL_EPOCHS=100
CKPT_DIR="${SCRATCH}/outputs/checkpoints/bridge_ts_v1"
LAST_CKPT="${CKPT_DIR}/latest.ckpt"
THIS_SCRIPT="${ROOT_DIR}/scripts/slurm/02_train_ts.slurm"
MASK_DIR="${SCRATCH}/Dataset/SEN12MSCRTS/cloud_masks"
NUM_WORKERS=10
PATH_OVERRIDES="paths.output_root=${SCRATCH}/outputs paths.bridge_ckpt_dir=${CKPT_DIR}"

if [[ "$(ls "${MASK_DIR}"/train/*_meta.npz 2>/dev/null | wc -l)" -eq 0 ]]; then
    echo "[ERROR] No cloud-mask index in ${MASK_DIR} — run 05_precompute_ts_clouds.slurm first"
    exit 1
fi

export TORCH_HOME="/scratch/katoch.aa/.cache/torch"
mkdir -p "${TORCH_HOME}/hub/checkpoints"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="offline"

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cloud-env
python -c "import omegaconf, lpips, rasterio" || { echo "[ERROR] cloud-env broken"; exit 1; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SAR Bridge Training — SEN12MS-CR-TS (T=3)"
echo "  Job ID : ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}"
echo "  Ckpts  : ${CKPT_DIR}"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! python -c "import torch; torch.cuda.init(); print(f'[INFO] PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
    echo "[WARN] Bad GPU driver on ${SLURMD_NODENAME} — resubmitting excluding this node"
    sbatch --exclude="${SLURMD_NODENAME}" "${THIS_SCRIPT}"
    exit 0
fi

CURRENT_EPOCH=0
if [[ -f "${LAST_CKPT}" ]]; then
    CURRENT_EPOCH=$(python - "${LAST_CKPT}" <<'PYEOF'
import torch, sys
try:
    print(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("epoch", 0))
except Exception:
    print(0)
PYEOF
)
fi
echo "[INFO] Current epoch from checkpoint: ${CURRENT_EPOCH} / ${TOTAL_EPOCHS}"

if [[ "${CURRENT_EPOCH}" -ge "${TOTAL_EPOCHS}" ]]; then
    echo "[INFO] Training complete — run evaluation with --config configs/sen12mscrts.yaml"
    exit 0
fi

NEXT_JOB=$(sbatch --dependency=afterany:"${SLURM_JOB_ID}" "${THIS_SCRIPT}" 2>&1)
echo "[INFO] Pre-scheduled next session: ${NEXT_JOB}"

if [[ -f "${LAST_CKPT}" ]]; then
    CKPT_ARG="--resume ${LAST_CKPT}"
    echo "[INFO] Resuming from ${LAST_CKPT}"
else
    CKPT_ARG=""
    echo "[INFO] Fresh training run"
fi

GPU_MEM=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // (1024**3))" 2>/dev/null || echo 0)
if [[ "${GPU_MEM}" -ge 60 ]]; then
    BATCH_ARGS="training.batch_size=8 training.accumulate_grad_batches=1"
else
    BATCH_ARGS="training.batch_size=2 training.accumulate_grad_batches=4"
fi
echo "[INFO] GPU memory: ${GPU_MEM}GB → ${BATCH_ARGS}"

PYTHONPATH="${ROOT_DIR}" python train/train_bridge.py \
    --config configs/sen12mscrts.yaml \
    training.num_workers="${NUM_WORKERS}" \
    training.num_epochs="${TOTAL_EPOCHS}" \
    logging.use_wandb=true \
    ${PATH_OVERRIDES} \
    ${BATCH_ARGS} \
    ${CKPT_ARG}

echo "[INFO] Session end: $(date)"
