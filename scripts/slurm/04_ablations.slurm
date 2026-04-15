#!/usr/bin/env bash
# =============================================================================
# 04_ablations.slurm — Run ablation studies (SLURM array job, 1 GPU per task)
# =============================================================================
# Five ablation studies, each submitted as an independent array task so they
# can run in parallel if the cluster has available GPUs.
#
#   Task 0 → backbone study  (3 training + 3 eval runs)
#   Task 1 → fusion study    (3 training + 3 eval runs)
#   Task 2 → loss study      (3 training + 3 eval runs)
#   Task 3 → nfe sweep       (eval only — reuses best.ckpt from main training)
#   Task 4 → schedule study  (3 training + 3 eval runs)
#
# Expected runtime per task:
#   backbone/fusion/loss/schedule : ~30-40 h  (3×50-epoch or 3×100-epoch runs)
#   nfe sweep                     : ~2 h      (eval only)
#
# Submit ALL five in parallel:
#   sbatch scripts/slurm/04_ablations.slurm
#
# Submit a single study (e.g. fusion only):
#   sbatch --array=1 scripts/slurm/04_ablations.slurm
#
# Study index mapping:
#   0=backbone  1=fusion  2=loss  3=nfe  4=schedule
#
# Run after main training (NFE study needs best.ckpt):
#   sbatch --dependency=afterok:<train_job_id> scripts/slurm/04_ablations.slurm
# =============================================================================
#SBATCH --job-name=sar_ablation
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=01:00:00
#SBATCH --array=0-4                    # 5 tasks (0..4); remove to run all as one job
#SBATCH --output=logs/slurm/ablation_%A_%a.out
#SBATCH --error=logs/slurm/ablation_%A_%a.err

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/slurm

# --- Environment ---------------------------------------------------------
module purge
module load anaconda3/2022.05           # check available: module avail anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cloud-env

echo "[INFO] Array job: ${SLURM_ARRAY_JOB_ID}  task: ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Node:      ${SLURMD_NODENAME}"
echo "[INFO] Start:     $(date)"

# --- Map task ID → study name -------------------------------------------
STUDIES=(backbone fusion loss nfe schedule)
STUDY="${STUDIES[${SLURM_ARRAY_TASK_ID}]}"
echo "[INFO] Running study: ${STUDY}"

# For the NFE sweep, point to the main training checkpoint
export BEST_CKPT="${BEST_CKPT:-${ROOT_DIR}/outputs/checkpoints/bridge/best.ckpt}"

# --- Dispatch to run_ablations.sh ----------------------------------------
DRY_RUN="${DRY_RUN:-0}" bash scripts/run_ablations.sh --study "${STUDY}"

echo "[INFO] Done:      $(date)"
echo "[INFO] Results in outputs/ablations/${STUDY}/"
