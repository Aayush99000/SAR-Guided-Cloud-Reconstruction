#!/bin/bash
# =============================================================================
# 08_eval_judge.slurm — Frozen-judge land-cover evaluation of a TS bridge
# =============================================================================
# First argument picks the experiment (v1 | tsar), as in 03_eval_ts.slurm.
# Uses our protocol (target excluded from the inputs) at NFE 3.
#   sbatch scripts/slurm/08_eval_judge.slurm v1
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
    *)    echo "[ERROR] unknown experiment '${EXP}' (expected v1 | tsar)"; exit 1 ;;
esac
CKPT="${CKPT:-${SCRATCH}/outputs/checkpoints/bridge_ts_${EXP}/best.ckpt}"
JUDGE="${JUDGE:-${SCRATCH}/outputs/checkpoints/judge/best.ckpt}"
WC_DIR="${SCRATCH}/Dataset/SEN12MSCRTS/worldcover"
NFE="${NFE:-3}"
LIMIT="${LIMIT:-0}"
if [[ "${LIMIT}" -gt 0 ]]; then
    OUT_DIR="${SCRATCH}/outputs/eval/judge_smoke_${EXP}"
else
    OUT_DIR="${SCRATCH}/outputs/eval/judge_ts_${EXP}"
fi
BACKUP_ROOT="/home/katoch.aa/SAR-Diffusion-Bridge/results/judge_ts_${EXP}"

CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cloud-env
export TORCH_HOME="/scratch/katoch.aa/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SCRATCH}"

echo "[INFO] Judge eval | experiment ${EXP} | job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} | $(date)"
echo "[INFO] Bridge ${CKPT}"
echo "[INFO] Judge  ${JUDGE}"
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
    data.ts.exclude_target_from_inputs=true

if [[ "${LIMIT}" -eq 0 ]]; then
    mkdir -p "${BACKUP_ROOT}"
    rsync -a "${OUT_DIR}/" "${BACKUP_ROOT}/" \
        && echo "[INFO] Results backed up to ${BACKUP_ROOT}" \
        || echo "[WARN] Backup to home failed (quota?)"
fi
echo "[INFO] Done: $(date)"
