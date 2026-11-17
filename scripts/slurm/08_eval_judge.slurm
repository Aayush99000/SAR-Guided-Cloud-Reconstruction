#!/bin/bash
# =============================================================================
# 08_eval_judge.slurm — Frozen-judge land-cover evaluation of a TS bridge
# =============================================================================
# First argument picks the experiment (v1 | tsar), as in 03_eval_ts.slurm.
# Uses our protocol (target excluded from the inputs) at NFE 3.
#   sbatch scripts/slurm/08_eval_judge.slurm v1
# Variant run (extra overrides, results under judge_ts_<exp>_<tag>):
#   sbatch --export=ALL,TAG=composite,EXTRA_ARGS=data.composite_output=true scripts/slurm/08_eval_judge.slurm v1
# Smoke test (64 samples, smoke judge):
#   sbatch --export=ALL,LIMIT=64,JUDGE=/scratch/.../judge_smoke/best.ckpt --time=00:30:00 \
#       scripts/slurm/08_eval_judge.slurm v1
# =============================================================================
#SBATCH --job-name=eval_judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/eval_judge_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/eval_judge_%j.err

set -euo pipefail

SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
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
CKPT="${CKPT:-${SCRATCH}/outputs/checkpoints/bridge_ts_${EXP}${RUN_SUFFIX}/best.ckpt}"
JUDGE="${JUDGE:-${SCRATCH}/outputs/checkpoints/judge/best.ckpt}"
WC_DIR="${SCRATCH}/Dataset/SEN12MSCRTS/worldcover"
NFE="${NFE:-3}"
LIMIT="${LIMIT:-0}"
TAG="${TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN="judge_ts_${EXP}${RUN_SUFFIX}${TAG:+_${TAG}}"
if [[ "${LIMIT}" -gt 0 ]]; then
    OUT_DIR="${SCRATCH}/outputs/eval/judge_smoke_${EXP}${RUN_SUFFIX}"
else
    OUT_DIR="${SCRATCH}/outputs/eval/${RUN}"
fi
BACKUP_ROOT="/home/katoch.aa/SAR-Diffusion-Bridge/results/${RUN}"

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cloud-env
export TORCH_HOME="/scratch/katoch.aa/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SCRATCH}"

echo "[INFO] Judge eval | experiment ${EXP} | job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} | $(date)"
echo "[INFO] Bridge ${CKPT}"
echo "[INFO] Judge  ${JUDGE}"
echo "[INFO] Extra  ${EXTRA_ARGS:-none}"
for f in "${CKPT}" "${JUDGE}"; do
    [[ -f "${f}" ]] || { echo "[ERROR] Missing ${f}"; exit 1; }
done

PYTHONPATH="${SCRATCH}" python train/evaluate_judge.py \
    --config ${EXTRA_CONFIGS} \
    --ckpt "${CKPT}" \
    --judge "${JUDGE}" \
    --wc-dir "${WC_DIR}" \
    --nfe "${NFE}" \
    --limit "${LIMIT}" \
    --out-dir "${OUT_DIR}" \
    training.num_workers=12 \
    data.ts.exclude_target_from_inputs=true ${EXTRA_ARGS}

if [[ "${LIMIT}" -eq 0 ]]; then
    mkdir -p "${BACKUP_ROOT}"
    rsync -a "${OUT_DIR}/" "${BACKUP_ROOT}/" \
        && echo "[INFO] Results backed up to ${BACKUP_ROOT}" \
        || echo "[WARN] Backup to home failed (quota?)"
fi
echo "[INFO] Done: $(date)"
