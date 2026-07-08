#!/usr/bin/env bash
#SBATCH --job-name=sar_bridge_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=d1026,d1028
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR}"
cd "${ROOT_DIR}"
mkdir -p logs/slurm

TOTAL_EPOCHS=100
SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
CKPT_DIR="${SCRATCH}/outputs/checkpoints/bridge_mt_v3"
LAST_CKPT="${CKPT_DIR}/latest.ckpt"
THIS_SCRIPT="${ROOT_DIR}/scripts/slurm/02_train_explorer.slurm"
NUM_WORKERS=8
DATA_ROOT="${SCRATCH}/Dataset/SEN12MSCR"
DATA_ROOT_OVERRIDE="paths.data_root=${DATA_ROOT} paths.output_root=${SCRATCH}/outputs paths.bridge_ckpt_dir=${CKPT_DIR}"

unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER

module purge
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cloud-env

export WANDB_API_KEY="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"
export WANDB_MODE="offline"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SAR Bridge Training (MT-SAR 8ch) — Explorer A100"
echo "  Job ID : ${SLURM_JOB_ID}"
echo "  Node   : ${SLURMD_NODENAME}"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- GPU health check: resubmit excluding ALL known bad nodes if driver broken ---
if ! python -c "import torch; torch.cuda.init(); print(f'[INFO] PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
    echo "[WARN] Bad GPU driver on ${SLURMD_NODENAME} — resubmitting excluding all bad nodes"
    # Must list ALL bad nodes here — command-line --exclude overrides #SBATCH --exclude
    sbatch --exclude="d1026,d1028,${SLURMD_NODENAME}" "${THIS_SCRIPT}"
    exit 0
fi

# --- Check if training is already complete ---
CURRENT_EPOCH=0
if [[ -f "${LAST_CKPT}" ]]; then
    CURRENT_EPOCH=$(PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" python - "${LAST_CKPT}" <<'PYEOF'
import torch, sys
try:
    ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except Exception:
    print(0)
PYEOF
)
fi

echo "[INFO] Current epoch from checkpoint: ${CURRENT_EPOCH} / ${TOTAL_EPOCHS}"

if [[ "${CURRENT_EPOCH}" -ge "${TOTAL_EPOCHS}" ]]; then
    echo "[INFO] Training complete — submitting eval"
    sbatch "${ROOT_DIR}/scripts/slurm/03_evaluate.slurm" 2>/dev/null || true
    exit 0
fi

# --- Pre-schedule next session NOW (before training starts) ---
# This runs regardless of how this job ends (time limit, crash, or success),
# because it is submitted before Python starts — not after it exits.
NEXT_JOB=$(sbatch --dependency=afterany:"${SLURM_JOB_ID}" \
           "${THIS_SCRIPT}" 2>&1)
echo "[INFO] Pre-scheduled next session: ${NEXT_JOB}"

# --- Resume logic ---
if [[ -n "${RESUME:-}" && -f "${RESUME}" ]]; then
    CKPT_ARG="--resume ${RESUME}"
    echo "[INFO] Resuming from env: ${RESUME}"
elif [[ -f "${LAST_CKPT}" ]]; then
    CKPT_ARG="--resume ${LAST_CKPT}"
    echo "[INFO] Resuming from disk: ${LAST_CKPT}"
else
    CKPT_ARG=""
    echo "[INFO] Fresh training run"
fi

# --- Detect GPU and set batch size accordingly ---
GPU_MEM=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // (1024**3))" 2>/dev/null || echo 0)
if [[ "${GPU_MEM}" -ge 60 ]]; then
    BATCH_ARGS="training.batch_size=8 training.accumulate_grad_batches=1"
else
    # V100 (32GB) — halve batch, double accumulation to keep effective batch=8
    BATCH_ARGS="training.batch_size=4 training.accumulate_grad_batches=2"
fi
echo "[INFO] GPU memory: ${GPU_MEM}GB → ${BATCH_ARGS}"

# --- Train ---
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" python train/train_bridge.py \
    --config configs/default.yaml \
    training.num_workers="${NUM_WORKERS}" \
    logging.use_wandb=true \
    ${DATA_ROOT_OVERRIDE} \
    ${BATCH_ARGS} \
    ${CKPT_ARG}

echo "[INFO] Session end: $(date)"
