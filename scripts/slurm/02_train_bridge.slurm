#!/usr/bin/env bash
# =============================================================================
# 02_train_bridge.slurm — Train the SAR-guided diffusion bridge (pixel-space)
# =============================================================================
# Single A100 (80 GB) job.  Trains SAROpticalUNet + OT-ODE bridge for 100
# epochs with EMA, AMP (bfloat16), AdamW, cosine-with-warmup LR schedule.
#
# Expected runtime: ~24-48 h on a single A100 for 100 epochs.
# Memory: ~28 GB GPU with batch_size=8, patch_size=256.
#
# Submit:
#   sbatch scripts/slurm/02_train_bridge.slurm
#
# Resume from checkpoint:
#   sbatch scripts/slurm/02_train_bridge.slurm --export=RESUME=outputs/checkpoints/bridge/last.ckpt
#
# Run as dependency (after download):
#   sbatch --dependency=afterok:<download_job_id> scripts/slurm/02_train_bridge.slurm
# =============================================================================
#SBATCH --job-name=sar_bridge_train
#SBATCH --partition=gpu                 # adjust: gpu / gpu-long / a100 / v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1              # adjust: a100 / v100 / gpu
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/slurm

# --- Environment ---------------------------------------------------------
module purge
# module load anaconda3/2023.03          # Northeastern Discovery
# module load cuda/11.8                  # if not bundled with conda env
conda activate cloud-env

# --- GPU info ------------------------------------------------------------
echo "[INFO] Job ID:  ${SLURM_JOB_ID}"
echo "[INFO] Node:    ${SLURMD_NODENAME}"
echo "[INFO] GPUs:    ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print(f'[INFO] PyTorch: {torch.__version__}  |  CUDA: {torch.cuda.get_device_name(0)}')"
echo "[INFO] Start:   $(date)"

# --- Build extra CLI args ------------------------------------------------
EXTRA_ARGS=""
if [[ -n "${RESUME:-}" ]]; then
    EXTRA_ARGS="--resume ${RESUME}"
    echo "[INFO] Resuming from: ${RESUME}"
fi

# Optional: override data root if your scratch path differs from paths.yaml
# EXTRA_ARGS="${EXTRA_ARGS} paths.data_root=/scratch/${USER}/SEN12MS-CR"

# --- Train ---------------------------------------------------------------
python train/train_bridge.py \
    --config configs/default.yaml \
    training.num_workers=8 \
    logging.use_wandb=true \
    ${EXTRA_ARGS}

echo "[INFO] Done:    $(date)"
