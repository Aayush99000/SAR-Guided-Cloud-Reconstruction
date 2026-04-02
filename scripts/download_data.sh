#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download and prepare the SEN12MS-CR dataset.
#
# SEN12MS-CR is hosted on the TUM mediaTUM repository.
# Paper: Ebel et al., "SEN12MS-CR: A Multi-Seasonal Dataset for
#         Cloud Removal in Multispectral Satellite Imagery", 2022.
#
# Usage:
#   bash scripts/download_data.sh [--dest /path/to/data]
# ---------------------------------------------------------------------------

set -euo pipefail

DEST="${1:-/data/SEN12MS-CR}"
mkdir -p "${DEST}"

echo "[INFO] Downloading SEN12MS-CR to ${DEST}"

# --- Base URLs (update if hosting changes) ---
BASE_URL="https://mediatum.ub.tum.de/download"

# Dataset parts (adjust IDs to match the actual mediaTUM entries)
PARTS=(
  "1554803"   # ROIs1158_spring
  "1554804"   # ROIs1158_summer
  "1554805"   # ROIs1158_fall
  "1554806"   # ROIs1158_winter
  "1554807"   # ROIs1868_spring
  "1554808"   # ROIs1868_summer
  "1554809"   # ROIs1868_fall
  "1554810"   # ROIs1868_winter
  "1554811"   # ROIs1970_spring
  "1554812"   # ROIs1970_summer
  "1554813"   # ROIs1970_fall
  "1554814"   # ROIs1970_winter
  "1554815"   # ROIs2017_spring
  "1554816"   # ROIs2017_summer
  "1554817"   # ROIs2017_fall
  "1554818"   # ROIs2017_winter
)

for PART_ID in "${PARTS[@]}"; do
  OUT_FILE="${DEST}/${PART_ID}.tar.gz"
  if [[ -f "${OUT_FILE}" ]]; then
    echo "[SKIP] ${OUT_FILE} already exists."
    continue
  fi
  echo "[DOWNLOAD] Part ${PART_ID} → ${OUT_FILE}"
  wget --progress=dot:mega \
       --retry-connrefused \
       --waitretry=5 \
       --tries=3 \
       -O "${OUT_FILE}" \
       "${BASE_URL}/${PART_ID}"
done

# --- Extract ---
echo "[INFO] Extracting archives..."
for PART_ID in "${PARTS[@]}"; do
  TAR="${DEST}/${PART_ID}.tar.gz"
  if [[ -f "${TAR}" ]]; then
    tar -xzf "${TAR}" -C "${DEST}"
    rm -f "${TAR}"
  fi
done

# --- Generate train/val/test splits ---
echo "[INFO] Generating split files..."
python - <<'EOF'
import os, random, pathlib

root = pathlib.Path("${DEST}")
scenes = sorted([d.name for d in root.iterdir() if d.is_dir()])
random.seed(42)
random.shuffle(scenes)

n = len(scenes)
train_end = int(0.7 * n)
val_end   = int(0.85 * n)

splits = {
    "train": scenes[:train_end],
    "val":   scenes[train_end:val_end],
    "test":  scenes[val_end:],
}

split_dir = root / "splits"
split_dir.mkdir(exist_ok=True)
for name, ids in splits.items():
    (split_dir / f"{name}.txt").write_text("\n".join(ids))
    print(f"  {name}: {len(ids)} scenes")
EOF

echo "[DONE] Dataset ready at ${DEST}"
