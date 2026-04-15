#!/usr/bin/env bash
# =============================================================================
# submit_pipeline.sh — Submit the full pipeline as chained SLURM jobs
# =============================================================================
#
# With the 8-hour wall-time limit the training stage uses CHAIN TRAINING:
# 02_train_bridge.slurm resubmits itself each session until all epochs are
# done, then automatically triggers eval (03) and ablations (04).
#
# Dependency graph:
#
#   01_download  →  02_train (session 1)
#                        ↓  resubmits itself
#                   02_train (session 2)
#                        ↓  ...
#                   02_train (final session)
#                        ├→  03_evaluate
#                        └→  04_ablations  [array, 5 tasks in parallel]
#
# Usage:
#   # Full pipeline (download + train chain + auto eval/ablations):
#   bash scripts/slurm/submit_pipeline.sh
#
#   # Skip download (data already on disk):
#   bash scripts/slurm/submit_pipeline.sh --skip-download
#
#   # Only eval + ablations (training done, best.ckpt exists):
#   bash scripts/slurm/submit_pipeline.sh --skip-download --skip-train
#
#   # Dry-run — print commands without submitting:
#   DRY_RUN=1 bash scripts/slurm/submit_pipeline.sh
#
# Logs: logs/slurm/<stage>_<jobid>.{out,err}
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

# --- Helper: submit and return job ID -------------------------------------
submit() {
    local dep_flag="${1:-}"
    local script="${2}"
    shift 2
    local extra="${*:-}"
    local cmd="sbatch ${dep_flag} ${script} ${extra}"
    echo "    \$ ${cmd}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "    [dry-run] → fake_job_$$"
        echo "fake_job_$$"
        return 0
    fi
    local out
    out=$(${cmd})
    echo "    → ${out}"
    echo "${out}" | awk '{print $NF}'
}

mkdir -p logs/slurm

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SAR Cloud Reconstruction — Pipeline Submission"
echo "  Wall-time limit: 8 h   Strategy: chain training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- Stage 1: Download --------------------------------------------------------
DOWNLOAD_DEP=""
if [[ "${SKIP_DOWNLOAD}" == "0" ]]; then
    echo ""
    echo "[1/2] Submitting data download..."
    DOWNLOAD_ID=$(submit "" "${SLURM_DIR}/01_download_data.slurm")
    DOWNLOAD_DEP="--dependency=afterok:${DOWNLOAD_ID}"
    echo "      Job ID: ${DOWNLOAD_ID}"
else
    echo "[1/2] Skipping download (data already on disk)"
fi

# --- Stage 2: Training chain --------------------------------------------------
if [[ "${SKIP_TRAIN}" == "0" ]]; then
    echo ""
    echo "[2/2] Submitting first training session..."
    echo "      (subsequent sessions and eval/ablations are triggered automatically)"
    TRAIN_ID=$(submit "${DOWNLOAD_DEP}" "${SLURM_DIR}/02_train_bridge.slurm")
    echo "      Job ID: ${TRAIN_ID}  ← first session only"
    echo ""
    echo "  After each 8-hour session, the script checks outputs/checkpoints/bridge/last.ckpt"
    echo "  and resubmits itself.  The final session submits eval and ablations."
else
    # Training already done — submit eval and ablations immediately
    CKPT="${CKPT:-${ROOT_DIR}/outputs/checkpoints/bridge/best.ckpt}"
    if [[ ! -f "${CKPT}" && "${DRY_RUN}" == "0" ]]; then
        echo "[ERROR] No checkpoint at ${CKPT}"
        echo "        Pass CKPT=<path> or run training first."
        exit 1
    fi
    echo "[2/2] Skipping training — submitting eval + ablations directly"
    echo ""
    echo "  Eval:"
    EVAL_ID=$(submit "" "${SLURM_DIR}/03_evaluate.slurm")
    echo "      Job ID: ${EVAL_ID}"
    echo ""
    echo "  Ablations:"
    ABL_ID=$(submit "" "${SLURM_DIR}/04_ablations.slurm")
    echo "      Job ID: ${ABL_ID}  (array 0-4)"
fi

# --- Summary ------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Monitor:"
echo "    squeue -u \$USER"
echo "    tail -f logs/slurm/train_<jobid>.out"
echo ""
echo "  Results:"
echo "    outputs/checkpoints/bridge/  ← last.ckpt + best.ckpt per session"
echo "    outputs/eval/                ← metrics CSVs + visualizations"
echo "    outputs/ablations/           ← 5 study outputs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
