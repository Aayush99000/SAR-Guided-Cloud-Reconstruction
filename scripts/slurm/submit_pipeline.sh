#!/usr/bin/env bash
# =============================================================================
# submit_pipeline.sh — Submit the full pipeline as chained SLURM jobs
# =============================================================================
# Creates a linear dependency chain:
#
#   download  →  train  →  evaluate  →  ablations
#
# Each stage only starts after the previous one completes successfully.
# The NFE ablation study (task 3) also depends on training finishing.
#
# Usage:
#   # Submit the full pipeline:
#   bash scripts/slurm/submit_pipeline.sh
#
#   # Skip data download (data already on disk):
#   bash scripts/slurm/submit_pipeline.sh --skip-download
#
#   # Skip download and training (only run eval + ablations):
#   bash scripts/slurm/submit_pipeline.sh --skip-download --skip-train
#
#   # Dry-run — print sbatch commands without submitting:
#   DRY_RUN=1 bash scripts/slurm/submit_pipeline.sh
#
# Output:
#   Logs are written to logs/slurm/<stage>_<jobid>.{out,err}
# =============================================================================

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SLURM_DIR="${ROOT_DIR}/scripts/slurm"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DOWNLOAD=0
SKIP_TRAIN=0

# --- Parse flags -----------------------------------------------------------
for arg in "$@"; do
    case "${arg}" in
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --skip-train)    SKIP_DOWNLOAD=1; SKIP_TRAIN=1 ;;
        *)               echo "[WARN] Unknown flag: ${arg}" ;;
    esac
done

# --- Helper ----------------------------------------------------------------
submit() {
    local dep_flag="${1:-}"
    local script="${2}"
    shift 2
    local cmd="sbatch ${dep_flag} ${script} $*"
    echo "  $ ${cmd}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  [dry-run] fake_job_id_$$"
        echo "fake_job_id_$$"          # stdout used for dependency chaining
        return 0
    fi
    # Submit and capture job ID from "Submitted batch job 12345"
    local out
    out=$(${cmd})
    echo "  → ${out}"
    echo "${out}" | awk '{print $NF}'  # job ID on stdout for chaining
}

mkdir -p logs/slurm

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SAR Cloud Reconstruction — Pipeline Submission"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- Stage 1: Download ---------------------------------------------------
DOWNLOAD_DEP=""
if [[ "${SKIP_DOWNLOAD}" == "0" ]]; then
    echo ""
    echo "[1/4] Submitting data download job..."
    DOWNLOAD_ID=$(submit "" "${SLURM_DIR}/01_download_data.slurm")
    DOWNLOAD_DEP="--dependency=afterok:${DOWNLOAD_ID}"
    echo "      Job ID: ${DOWNLOAD_ID}"
else
    echo "[1/4] Skipping download (--skip-download)"
fi

# --- Stage 2: Train bridge -----------------------------------------------
TRAIN_DEP=""
if [[ "${SKIP_TRAIN}" == "0" ]]; then
    echo ""
    echo "[2/4] Submitting bridge training job..."
    TRAIN_ID=$(submit "${DOWNLOAD_DEP}" "${SLURM_DIR}/02_train_bridge.slurm")
    TRAIN_DEP="--dependency=afterok:${TRAIN_ID}"
    echo "      Job ID: ${TRAIN_ID}"
else
    echo "[2/4] Skipping training (--skip-train)"
    # eval and ablations still need a checkpoint
    CKPT="${CKPT:-${ROOT_DIR}/outputs/checkpoints/bridge/best.ckpt}"
    if [[ ! -f "${CKPT}" && "${DRY_RUN}" == "0" ]]; then
        echo "[ERROR] No checkpoint found at ${CKPT}"
        echo "        Pass CKPT=<path> or run training first."
        exit 1
    fi
fi

# --- Stage 3: Evaluate ---------------------------------------------------
echo ""
echo "[3/4] Submitting evaluation job..."
EVAL_ID=$(submit "${TRAIN_DEP}" "${SLURM_DIR}/03_evaluate.slurm")
echo "      Job ID: ${EVAL_ID}"

# --- Stage 4: Ablation studies (array, all 5 in parallel) ----------------
echo ""
echo "[4/4] Submitting ablation array job (5 tasks)..."
# Ablations depend on training (need checkpoint); run concurrently with eval
ABL_ID=$(submit "${TRAIN_DEP}" "${SLURM_DIR}/04_ablations.slurm")
echo "      Job ID: ${ABL_ID}  (array: tasks 0-4)"

# --- Summary -------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Submitted jobs:"
[[ "${SKIP_DOWNLOAD}" == "0" ]] && echo "    Download   → ${DOWNLOAD_ID:-dry-run}"
[[ "${SKIP_TRAIN}"    == "0" ]] && echo "    Train      → ${TRAIN_ID:-dry-run}"
echo "    Evaluate   → ${EVAL_ID:-dry-run}"
echo "    Ablations  → ${ABL_ID:-dry-run}  [array 0-4]"
echo ""
echo "  Monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f logs/slurm/train_<jobid>.out"
echo ""
echo "  Results will appear in:"
echo "    outputs/checkpoints/bridge/    ← training checkpoints"
echo "    outputs/eval/                  ← evaluation metrics + visualizations"
echo "    outputs/ablations/             ← ablation study outputs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
