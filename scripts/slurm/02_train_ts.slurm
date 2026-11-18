#!/usr/bin/env bash
# =============================================================================
# 02_train_ts.slurm — Train the OT-ODE bridge on SEN12MS-CR-TS
# =============================================================================
# Self-chaining 8 h sessions; resumes from latest.ckpt. First argument picks the
# experiment (checkpoints go to outputs/checkpoints/bridge_ts_<exp>):
#   sbatch -J bridge_ts_v1   scripts/slurm/02_train_ts.slurm v1    # 3 input dates' SAR (6ch)
#   sbatch -J bridge_ts_tsar scripts/slurm/02_train_ts.slurm tsar  # + target-date SAR (8ch)
#   sbatch -J bridge_ts_loss scripts/slurm/02_train_ts.slurm loss  # v1 with a uniform loss (alpha 0.5)
#   sbatch -J bridge_ts_mask scripts/slurm/02_train_ts.slurm mask  # v1 + cloud mask as network input
#   sbatch -J bridge_ts_temporal_a08 scripts/slurm/02_train_ts.slurm temporal_a08  # mask + attention over input dates
# loss / temporal (= temporal_a08 with alpha 0.5) pin the uniform loss; every other
# experiment uses the cloud-weighted loss (alpha 0.8) from configs/sen12mscrts.yaml.
#   (tsar_mask = target-date SAR + mask; temporal_concat = mask + concatenated dates;
#    tsar_temporal = target-date SAR + temporal;
#    tsar_temporal_change = tsar_temporal + explicit SAR change gating)
# Optional second argument: official (default, checkpoints in bridge_ts_<exp>_official)
# or legacy (the pre-Nov-2026 runs, which trained on copies of three test tiles).
# Requires the cloud-mask index from 05_precompute_ts_clouds.slurm.
# =============================================================================
#SBATCH --job-name=bridge_ts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# SXM2 V100 nodes only: ~1.6 s/step there vs ~4.3 s on the T4 and ~7 s on the
# PCIe V100 (EPYC 7351) nodes, where the data loader stalls the GPU
#SBATCH --gres=gpu:v100-sxm2:1
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

EXP="${1:-v1}"
case "${EXP}" in
    v1)   EXTRA_CONFIGS="configs/sen12mscrts.yaml" ;;
    tsar) EXTRA_CONFIGS="configs/sen12mscrts.yaml configs/sen12mscrts_tsar.yaml" ;;
    # loss / temporal were trained with the uniform loss (alpha 0.5): keep it
    loss) EXTRA_CONFIGS="configs/sen12mscrts.yaml loss.alpha=0.5" ;;
    mask) EXTRA_CONFIGS="configs/sen12mscrts.yaml configs/sen12mscrts_mask.yaml" ;;
    tsar_mask) EXTRA_CONFIGS="configs/sen12mscrts.yaml configs/sen12mscrts_tsar.yaml configs/sen12mscrts_mask.yaml" ;;
    temporal) EXTRA_CONFIGS="configs/sen12mscrts.yaml configs/sen12mscrts_temporal.yaml loss.alpha=0.5" ;;
    temporal_a08) EXTRA_CONFIGS="configs/sen12mscrts.yaml configs/sen12mscrts_temporal.yaml" ;;
    temporal_concat) EXTRA_CONFIGS="configs/sen12mscrts.yaml configs/sen12mscrts_temporal_concat.yaml" ;;
    tsar_temporal) EXTRA_CONFIGS="configs/sen12mscrts.yaml configs/sen12mscrts_tsar.yaml configs/sen12mscrts_temporal.yaml" ;;
    tsar_temporal_change) EXTRA_CONFIGS="configs/sen12mscrts.yaml configs/sen12mscrts_tsar.yaml configs/sen12mscrts_temporal.yaml configs/sen12mscrts_sar_change.yaml" ;;
    *)    echo "[ERROR] unknown experiment '${EXP}' (expected v1 | tsar | loss | mask | tsar_mask | temporal | temporal_a08 | temporal_concat | tsar_temporal | tsar_temporal_change)"; exit 1 ;;
esac
# Second argument: official (default; UnCRtainTS split, runs named *_official) or
# legacy (the pre-Nov-2026 split that trained on copies of three test tiles).
SPLIT="${2:-official}"
case "${SPLIT}" in
    official) RUN_SUFFIX="_official" ;;
    legacy)   RUN_SUFFIX=""; EXTRA_CONFIGS="${EXTRA_CONFIGS} data.ts.split_protocol=legacy" ;;
    *)        echo "[ERROR] unknown split '${SPLIT}' (expected official | legacy)"; exit 1 ;;
esac

TOTAL_EPOCHS=100
NAME="bridge_ts_${EXP}${RUN_SUFFIX}"
CKPT_DIR="${SCRATCH}/outputs/checkpoints/${NAME}"
LAST_CKPT="${CKPT_DIR}/latest.ckpt"
THIS_SCRIPT="${ROOT_DIR}/scripts/slurm/02_train_ts.slurm"
MASK_DIR="${SCRATCH}/Dataset/SEN12MSCRTS/cloud_masks"
NUM_WORKERS=12
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
echo "  SAR Bridge Training — SEN12MS-CR-TS (T=3) — experiment: ${EXP}  split: ${SPLIT}"
echo "  Configs: ${EXTRA_CONFIGS}"
echo "  Job ID : ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}"
echo "  Ckpts  : ${CKPT_DIR}"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! python -c "import torch; torch.cuda.init(); print(f'[INFO] PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
    echo "[WARN] Bad GPU driver on ${SLURMD_NODENAME} — resubmitting excluding this node"
    sbatch -J "${NAME}" --exclude="${SLURMD_NODENAME}" "${THIS_SCRIPT}" "${EXP}" "${SPLIT}"
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
    echo "[INFO] Training complete — submitting test evaluation"
    if [[ -f "${CKPT_DIR}/.eval_submitted" ]]; then
        echo "[INFO] Evaluation already submitted ($(cat "${CKPT_DIR}/.eval_submitted"))"
    else
        # Retry while the submit limit is full (an array of 2 needs 2 free slots)
        for attempt in $(seq 1 120); do
            if EVAL_JOB=$(sbatch -J "eval_ts_${EXP}${RUN_SUFFIX}" "${ROOT_DIR}/scripts/slurm/03_eval_ts.slurm" "${EXP}" "${SPLIT}" 2>&1); then
                echo "${EVAL_JOB}" | tee "${CKPT_DIR}/.eval_submitted"
                break
            fi
            echo "[WARN] Could not submit the evaluation (attempt ${attempt}): ${EVAL_JOB}"
            sleep 60
        done
    fi
    exit 0
fi

# Pre-schedule the next session. sbatch fails while the per-user submit limit
# (8 on gpu) is full, which used to end the chain under set -e; retry in the
# background for the whole session (other projects can hold the limit for hours).
(
    for attempt in $(seq 1 460); do
        if NEXT_JOB=$(sbatch -J "${NAME}" --dependency=afterany:"${SLURM_JOB_ID}" "${THIS_SCRIPT}" "${EXP}" "${SPLIT}" 2>&1); then
            echo "[INFO] Pre-scheduled next session: ${NEXT_JOB}"
            exit 0
        fi
        echo "[WARN] Could not pre-schedule the next session (attempt ${attempt}): ${NEXT_JOB}"
        sleep 60
    done
    echo "[ERROR] Next session never scheduled; resubmit by hand: sbatch -J ${NAME} ${THIS_SCRIPT} ${EXP} ${SPLIT}"
) &

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
    --config ${EXTRA_CONFIGS} \
    training.num_workers="${NUM_WORKERS}" \
    training.num_epochs="${TOTAL_EPOCHS}" \
    logging.use_wandb=true \
    ${PATH_OVERRIDES} \
    ${BATCH_ARGS} \
    ${CKPT_ARG}

echo "[INFO] Session end: $(date)"
