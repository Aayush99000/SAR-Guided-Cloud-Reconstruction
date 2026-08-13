#!/usr/bin/env bash
#SBATCH --job-name=sen12ms_fix3
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/fix_missing_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/fix_missing_%j.err

# No global set -e — handle each archive independently so one failure doesn't kill the rest
set -uo pipefail

DEST="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/Dataset"
MSCR="${DEST}/SEN12MSCR"
TOKEN="m1554803"
FAILED_ARCHIVES=()

echo "=== SEN12MS-CR Missing Archives Fix (v3) ==="
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}  Start: $(date)"

# Download + verify + extract one archive; sets target_dir subdir
_handle_archive() {
    local filename="$1"
    local target="$2"   # s1, s2, or s2_cloudy
    local dest="${DEST}/${filename}"
    local tmpdir="${DEST}/tmp_${filename%.tar.gz}"

    echo ""
    echo "=== ${filename} ==="

    # Download with wget (supports resume via -c)
    local url1="https://dataserv.ub.tum.de/index.php/s/${TOKEN}/download?path=/&files=${filename}"
    local url2="https://dataserv.ub.tum.de/public.php/dav/files/${TOKEN}/${filename}"

    local ok=0
    for url in "${url1}" "${url2}"; do
        echo "[DL] ${url%%\?*}..."
        wget -q --show-progress \
             -c \
             --tries=5 --waitretry=30 --timeout=300 \
             --user-agent="Mozilla/5.0" \
             -O "${dest}" "${url}" 2>&1 && ok=1 && break || true
        local size; size=$(wc -c < "${dest}" 2>/dev/null || echo 0)
        if (( size >= 104857600 )); then ok=1; break; fi   # >100MB = likely real file
        echo "[WARN] Only ${size} bytes, trying next URL"
        rm -f "${dest}"
    done

    if [[ $ok -eq 0 ]]; then
        echo "[ERROR] Download failed for ${filename} — skipping"
        FAILED_ARCHIVES+=("${filename}")
        return 1
    fi

    local size; size=$(wc -c < "${dest}" 2>/dev/null || echo 0)
    echo "[OK] ${filename}: ${size} bytes"

    # Test archive integrity before extracting
    echo "[VERIFY] Testing archive integrity..."
    if ! gzip -t "${dest}" 2>/dev/null; then
        echo "[WARN] Archive integrity check failed — attempting partial extract anyway"
    fi

    # Extract (tolerate EOF errors for partial archives)
    rm -rf "${tmpdir}" && mkdir -p "${tmpdir}"
    echo "[EXTRACT] Extracting to ${tmpdir}..."
    tar --extract --file "${dest}" -C "${tmpdir}" 2>/dev/null || true

    local extracted; extracted=$(find "${tmpdir}" -name "*.tif" | wc -l)
    echo "[INFO] Extracted ${extracted} .tif files"

    if (( extracted == 0 )); then
        echo "[ERROR] No .tif files extracted from ${filename}"
        FAILED_ARCHIVES+=("${filename}")
        rm -rf "${tmpdir}" "${dest}"
        return 1
    fi

    # Move to flat target directory
    find "${tmpdir}" -name "*.tif" -exec mv {} "${MSCR}/${target}/" \;
    rm -rf "${tmpdir}" "${dest}"
    echo "[DONE] ${filename}: ${extracted} files → ${MSCR}/${target}/"
}

# --- Spring S1 (re-download, clean up partials first) ---
echo ""
echo "--- Cleaning up corrupted spring s1 ---"
rm -f "${DEST}/ROIs1158_spring_s1.tar.gz"
find "${MSCR}/s1/" -name "*spring*" -delete 2>/dev/null || true
rm -rf "${DEST}/ROIs1158_spring_s1" 2>/dev/null || true

_handle_archive "ROIs1158_spring_s1.tar.gz" "s1" || true

# --- 4 archives that were never downloaded ---
_handle_archive "ROIs1158_spring_s2_cloudy.tar.gz" "s2_cloudy" || true
_handle_archive "ROIs1868_summer_s1.tar.gz"         "s1"       || true
_handle_archive "ROIs1868_summer_s2.tar.gz"         "s2"       || true
_handle_archive "ROIs1970_fall_s2.tar.gz"           "s2"       || true

echo ""
echo "=== Final counts ==="
echo "  s1:        $(ls ${MSCR}/s1/        | wc -l)"
echo "  s2:        $(ls ${MSCR}/s2/        | wc -l)"
echo "  s2_cloudy: $(ls ${MSCR}/s2_cloudy/ | wc -l)"

if (( ${#FAILED_ARCHIVES[@]} > 0 )); then
    echo ""
    echo "[WARN] Failed archives: ${FAILED_ARCHIVES[*]}"
fi

# --- Regenerate splits ---
echo ""
echo "[INFO] Regenerating splits..."
module purge
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cloud-env

python3 - "${MSCR}" <<'PYEOF'
import sys, csv, random, pathlib
mscr = pathlib.Path(sys.argv[1])
s2_dir, s2c_dir, s1_dir = mscr/'s2', mscr/'s2_cloudy', mscr/'s1'
splits_dir = mscr/'splits'; splits_dir.mkdir(exist_ok=True)
records = []
for s2_path in sorted(s2_dir.glob('*.tif')):
    stem = s2_path.stem
    s2c_name = stem.replace('_s2_', '_s2_cloudy_', 1) + '.tif'
    s1_name  = stem.replace('_s2_', '_s1_', 1) + '.tif'
    if not (s2c_dir/s2c_name).exists(): continue
    if not (s1_dir/s1_name).exists():   continue
    parts = stem.split('_')
    records.append({'patch_id': f'{parts[0]}_{parts[1]}_{parts[3]}_{parts[4]}',
        's1': f's1/{s1_name}', 's2_clean': f's2/{s2_path.name}',
        's2_cloudy': f's2_cloudy/{s2c_name}', 'season': parts[1], 'roi': parts[0]})
random.seed(42); random.shuffle(records)
n = len(records); n_train = int(0.70*n); n_val = int(0.15*n)
for name, rows in [('train',records[:n_train]),('val',records[n_train:n_train+n_val]),('test',records[n_train+n_val:])]:
    with open(splits_dir/f'{name}.csv','w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['patch_id','s1','s2_clean','s2_cloudy','season','roi'])
        w.writeheader(); w.writerows(rows)
    print(f'  {name:5s}: {len(rows):6d} patches')
print(f'  Total: {n} complete triplets')
PYEOF

echo "Done: $(date)"
