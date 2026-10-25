#!/bin/bash
# =============================================================================
# 03_eval_ts.slurm — Evaluate the SEN12MS-CR-TS bridge on the official test split
# =============================================================================
# Array task 0 = "ours":       target excluded from the inputs (default config)
# Array task 1 = "uncrtaints": target may be one of the inputs, reproducing the
#                              UnCRtainTS fixed_sampler with min_cov=0 / max_cov=1
# Both use eval_target_policy=least_cloudy, eval_input_policy=first.
#
# Submit (from anywhere):
#   sbatch scripts/slurm/03_eval_ts.slurm
# Override checkpoint:
#   sbatch --export=ALL,CKPT=/path/to/epoch_0100.ckpt scripts/slurm/03_eval_ts.slurm
# =============================================================================
#SBATCH --job-name=eval_ts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/eval_ts_%A_%a.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/eval_ts_%A_%a.err

set -euo pipefail

SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
CKPT="${CKPT:-${SCRATCH}/outputs/checkpoints/bridge_ts_v1/best.ckpt}"
BACKUP_ROOT="/home/katoch.aa/SAR-Diffusion-Bridge/results/eval_ts_v1"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) PROTOCOL="ours";       PROTO_ARGS="data.ts.exclude_target_from_inputs=true" ;;
    1) PROTOCOL="uncrtaints"; PROTO_ARGS="data.ts.exclude_target_from_inputs=false" ;;
    *) echo "[ERROR] unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
OUT_DIR="${SCRATCH}/outputs/eval/ts_v1/${PROTOCOL}"

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cloud-env
export TORCH_HOME="/scratch/katoch.aa/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SCRATCH}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SEN12MS-CR-TS test evaluation — protocol: ${PROTOCOL}"
echo "  Job ID : ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}"
echo "  Ckpt   : ${CKPT}"
echo "  Out    : ${OUT_DIR}"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -c "import torch; print(f'[INFO] PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)}')"

if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    exit 1
fi

PYTHONPATH="${SCRATCH}" python train/evaluate.py \
    --config configs/sen12mscrts.yaml \
    --ckpt "${CKPT}" \
    --nfe 1 3 5 10 \
    --viz-every 50 \
    --lpips \
    --out-dir "${OUT_DIR}" \
    training.num_workers=12 \
    ${PROTO_ARGS}

# Scratch can be purged; back up the small result files (not the PNG strips).
mkdir -p "${BACKUP_ROOT}/${PROTOCOL}"
rsync -a --exclude='visualizations/' "${OUT_DIR}/" "${BACKUP_ROOT}/${PROTOCOL}/" \
    && echo "[INFO] Metrics backed up to ${BACKUP_ROOT}/${PROTOCOL}" \
    || echo "[WARN] Backup to home failed (quota?) — results remain in ${OUT_DIR}"

echo "[INFO] Done: $(date)"
