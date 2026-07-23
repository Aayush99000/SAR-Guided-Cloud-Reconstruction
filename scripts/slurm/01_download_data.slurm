#!/usr/bin/env bash
# =============================================================================
# 01_download_data.slurm — Download & reorganise SEN12MS-CR on Explorer HPC
# =============================================================================
# Downloads: SEN12MS-CR (monotemporal) — S2 cloud-free, S2 cloudy, S1 SAR
# All 4 seasons (spring/summer/fall/winter), ~300 GB total.
# No GPU needed. Typical wall time: 4-8 h depending on TUM server load.
#
# Submit:
#   sbatch scripts/slurm/01_download_data.slurm
# =============================================================================
#SBATCH --job-name=sen12ms_dl
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/download_%j.out
#SBATCH --error=logs/slurm/download_%j.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR}"
cd "${ROOT_DIR}"
mkdir -p logs/slurm

SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
DEST="${SCRATCH}/Dataset"
MSCR="${DEST}/SEN12MSCR"

mkdir -p "${MSCR}/s1" "${MSCR}/s2" "${MSCR}/s2_cloudy" "${MSCR}/splits"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SEN12MS-CR Download"
echo "  Job ID : ${SLURM_JOB_ID}"
echo "  Node   : ${SLURMD_NODENAME}"
echo "  Target : ${MSCR}"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ---------------------------------------------------------------------------
# Download helper — 3-strategy fallback (NextCloud index.php → WebDAV → /s/)
# ---------------------------------------------------------------------------
_dl() {
    local token="$1" filename="$2" destdir="$3"
    local dest="${destdir}/${filename}"

    if [[ -f "${dest}" ]] && (( $(wc -c < "${dest}") > 1024 )); then
        echo "[SKIP]     ${filename} (archive already present)"
        return 0
    fi

    local urls=(
        "https://dataserv.ub.tum.de/index.php/s/${token}/download?path=/&files=${filename}"
        "https://dataserv.ub.tum.de/public.php/dav/files/${token}/${filename}"
        "https://dataserv.ub.tum.de/s/${token}/download?path=/&files=${filename}"
    )

    local size=0
    for url in "${urls[@]}"; do
        echo "[DOWNLOAD] ${filename}"
        curl -L -k -u "${token}:" \
             --retry 5 --retry-delay 15 --retry-max-time 300 \
             --connect-timeout 60 --max-time 7200 \
             --user-agent "Mozilla/5.0" \
             --progress-bar \
             -o "${dest}" "${url}" 2>&1 || true
        size=$(wc -c < "${dest}" 2>/dev/null || echo 0)
        if (( size >= 1024 )); then break; fi
        echo "  → Only ${size} bytes, trying next URL..."
        rm -f "${dest}"
    done

    if (( size < 1024 )); then
        echo "[ERROR] All strategies failed for ${filename}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Extract helper
# ---------------------------------------------------------------------------
_extract() {
    local archive="$1" destdir="$2"
    echo "[EXTRACT]  ${archive##*/}"
    tar --extract --file "${archive}" -C "${destdir}"
    rm -f "${archive}"
}

# ---------------------------------------------------------------------------
# Files to download  (token m1554803 = SEN12MS-CR monotemporal)
# ---------------------------------------------------------------------------
TOKEN="m1554803"

declare -A FILES
FILES=(
    # S2 cloud-free
    [ROIs1158_spring_s2.tar.gz]="${DEST}"
    [ROIs1868_summer_s2.tar.gz]="${DEST}"
    [ROIs1970_fall_s2.tar.gz]="${DEST}"
    [ROIs2017_winter_s2.tar.gz]="${DEST}"
    # S2 cloudy
    [ROIs1158_spring_s2_cloudy.tar.gz]="${DEST}"
    [ROIs1868_summer_s2_cloudy.tar.gz]="${DEST}"
    [ROIs1970_fall_s2_cloudy.tar.gz]="${DEST}"
    [ROIs2017_winter_s2_cloudy.tar.gz]="${DEST}"
    # S1 SAR
    [ROIs1158_spring_s1.tar.gz]="${DEST}"
    [ROIs1868_summer_s1.tar.gz]="${DEST}"
    [ROIs1970_fall_s1.tar.gz]="${DEST}"
    [ROIs2017_winter_s1.tar.gz]="${DEST}"
)

for filename in "${!FILES[@]}"; do
    destdir="${FILES[$filename]}"
    dirname="${filename%.tar.gz}"
    extracted_dir="${destdir}/${dirname}"

    if [[ -d "${extracted_dir}" ]]; then
        echo "[SKIP]     ${filename} (already extracted)"
        continue
    fi

    _dl "${TOKEN}" "${filename}" "${destdir}"
    _extract "${destdir}/${filename}" "${destdir}"
done

# ---------------------------------------------------------------------------
# Flatten into s1/ s2/ s2_cloudy/
# ---------------------------------------------------------------------------
echo "[INFO] Flattening into s1/ s2/ s2_cloudy/ ..."

# S2 cloud-free
find "${DEST}" -maxdepth 4 -path "*/ROIs*_s2/*/*.tif" ! -path "*cloudy*" \
    -exec mv {} "${MSCR}/s2/" \; 2>/dev/null || true

# S2 cloudy
find "${DEST}" -maxdepth 4 -path "*/ROIs*_s2_cloudy/*/*.tif" \
    -exec mv {} "${MSCR}/s2_cloudy/" \; 2>/dev/null || true

# S1 SAR
find "${DEST}" -maxdepth 4 -path "*/ROIs*_s1/*/*.tif" \
    -exec mv {} "${MSCR}/s1/" \; 2>/dev/null || true

echo "[INFO] File counts:"
echo "  s1/        $(ls ${MSCR}/s1/       | wc -l) files"
echo "  s2/        $(ls ${MSCR}/s2/       | wc -l) files"
echo "  s2_cloudy/ $(ls ${MSCR}/s2_cloudy/| wc -l) files"

# ---------------------------------------------------------------------------
# Generate train/val/test CSV splits (70/15/15)
# ---------------------------------------------------------------------------
echo "[INFO] Generating splits..."

module purge
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cloud-env

python3 - "${MSCR}" <<'PYEOF'
import sys, csv, random, pathlib

mscr     = pathlib.Path(sys.argv[1])
s2_dir   = mscr / "s2"
s2c_dir  = mscr / "s2_cloudy"
s1_dir   = mscr / "s1"
splits   = mscr / "splits"
splits.mkdir(exist_ok=True)

records = []
for s2_path in sorted(s2_dir.glob("*.tif")):
    stem    = s2_path.stem
    s2c_name = stem.replace("_s2_", "_s2_cloudy_", 1) + ".tif"
    s1_name  = stem.replace("_s2_", "_s1_", 1) + ".tif"
    if not (s2c_dir / s2c_name).exists(): continue
    if not (s1_dir  / s1_name).exists():  continue
    parts  = stem.split("_")
    roi, season = parts[0], parts[1]
    tile, patch = parts[3], parts[4]
    records.append({
        "patch_id":  f"{roi}_{season}_{tile}_{patch}",
        "s1":        f"s1/{s1_name}",
        "s2_clean":  f"s2/{s2_path.name}",
        "s2_cloudy": f"s2_cloudy/{s2c_name}",
        "season":    season,
        "roi":       roi,
    })

random.seed(42)
random.shuffle(records)
n = len(records)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
split_map = {
    "train": records[:n_train],
    "val":   records[n_train:n_train+n_val],
    "test":  records[n_train+n_val:],
}
fields = ["patch_id","s1","s2_clean","s2_cloudy","season","roi"]
for name, rows in split_map.items():
    with open(splits / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  {name:5s}: {len(rows):6d} patches")
print(f"  Total: {n} complete triplets")
PYEOF

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Dataset ready at: ${MSCR}"
echo "  Done: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
