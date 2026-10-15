#!/bin/bash
#SBATCH --job-name=eval_v3lpips
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/eval_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/eval_%j.err

# ── Clean environment (prevents "Argument list too long" from env var bloat) ──
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

# ── Conda (direct source — module system unavailable after exec env -i) ──
CONDA_BASE="/shared/EL9/explorer/anaconda3/2024.06"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cloud-env

# ── Paths ──
SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
HOME_CKPT="/home/katoch.aa/SAR-Diffusion-Bridge/checkpoints/best.ckpt"
OUT_DIR="${SCRATCH}/outputs/eval/v3_lpips"

export TORCH_HOME="/scratch/katoch.aa/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${SCRATCH}/logs/slurm"
mkdir -p "${OUT_DIR}"

cd "${SCRATCH}"

echo "[$(date)] Starting v3+LPIPS evaluation on node ${SLURMD_NODENAME:-unknown}"
echo "[$(date)] Checkpoint: ${HOME_CKPT}"
echo "[$(date)] Output dir: ${OUT_DIR}"

python train/evaluate.py \
    --ckpt  "${HOME_CKPT}" \
    --nfe   1 \
    --viz-every 20 \
    --lpips \
    --out-dir "${OUT_DIR}" \
    --batch-size 4

EXIT_CODE=$?
echo "[$(date)] Evaluation finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
