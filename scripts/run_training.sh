#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end training pipeline:
#   Stage 1 — Train VQ-GAN
#   Stage 2 — Train Diffusion Bridge (with frozen VQ-GAN)
#   Stage 3 — Evaluate on test split
#
# Usage:
#   bash scripts/run_training.sh [--stage 1|2|3|all]
#
# Requires: conda environment or venv with requirements installed.
# ---------------------------------------------------------------------------

set -euo pipefail

STAGE="${1:-all}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Activate virtual environment if present
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[INFO] Using conda env: ${CONDA_DEFAULT_ENV}"
fi

PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run_stage() {
  local name="$1"
  local script="$2"
  shift 2
  echo ""
  echo "============================================================"
  echo "  ${name}"
  echo "============================================================"
  ${PYTHON} "${script}" "$@"
}

# ---------------------------------------------------------------------------
# Stage 1: VQ-GAN
# ---------------------------------------------------------------------------
if [[ "${STAGE}" == "1" || "${STAGE}" == "all" ]]; then
  run_stage "Stage 1: VQ-GAN Training" train/train_vqgan.py \
    training.num_epochs=50 \
    training.batch_size=8 \
    vqgan.latent_dim=256
fi

# ---------------------------------------------------------------------------
# Stage 2: Diffusion Bridge
# ---------------------------------------------------------------------------
if [[ "${STAGE}" == "2" || "${STAGE}" == "all" ]]; then
  run_stage "Stage 2: Diffusion Bridge Training" train/train_bridge.py \
    training.num_epochs=100 \
    training.batch_size=4 \
    diffusion.diffusion_steps=1000 \
    diffusion.alpha_schedule_type=sine \
    diffusion.sampler_nfe=5
fi

# ---------------------------------------------------------------------------
# Stage 3: Evaluation
# ---------------------------------------------------------------------------
if [[ "${STAGE}" == "3" || "${STAGE}" == "all" ]]; then
  run_stage "Stage 3: Evaluation" train/evaluate.py
fi

echo ""
echo "[DONE] Pipeline complete."
