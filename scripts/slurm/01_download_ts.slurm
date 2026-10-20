#!/bin/bash
# =============================================================================
# 01_download_ts.slurm — Download SEN12MS-CR-TS on Explorer HPC
# =============================================================================
# Downloads the multitemporal dataset (Ebel et al. 2022) — all 5 regions,
# S2 cloud-free + S2 cloudy + S1 SAR, train + test splits.
# Tokens: m1639953 (train), m1659251 (test)
# Approx size: ~600 GB compressed, ~1.5 TB extracted
#
# Submit:
#   sbatch scripts/slurm/01_download_ts.slurm
# =============================================================================
#SBATCH --job-name=dl_sen12ts
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/dl_ts_%j.out
#SBATCH --error=/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/logs/slurm/dl_ts_%j.err

set -euo pipefail

SCRATCH="/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction"
DEST="${SCRATCH}/Dataset/SEN12MSCRTS"
LOGS="${SCRATCH}/logs/slurm"

mkdir -p "${DEST}" "${LOGS}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SEN12MS-CR-TS Download"
echo "  Job ID : ${SLURM_JOB_ID}"
echo "  Node   : ${SLURMD_NODENAME}"
echo "  Target : ${DEST}"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ---------------------------------------------------------------------------
# Download helper — 3-strategy fallback
# ---------------------------------------------------------------------------
_dl() {
    local token="$1" filename="$2" destfile="$3"

    if [[ -f "${destfile}" ]] && (( $(wc -c < "${destfile}") > 1048576 )); then
        echo "[SKIP]     ${filename} (archive present)"
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
             --retry 5 --retry-delay 15 --retry-max-time 600 \
             --connect-timeout 60 --max-time 14400 \
             --user-agent "Mozilla/5.0" \
             --progress-bar \
             -o "${destfile}" "${url}" 2>&1 || true
        size=$(wc -c < "${destfile}" 2>/dev/null || echo 0)
        if (( size > 1048576 )); then break; fi
        echo "  → Only ${size} bytes, trying next URL..."
        rm -f "${destfile}"
    done

    if (( size <= 1048576 )); then
        echo "[ERROR] All strategies failed for ${filename}"
        exit 1
    fi
}

_extract() {
    local archive="$1" destdir="$2"
    echo "[EXTRACT]  ${archive##*/} → ${destdir}"
    tar --extract --file "${archive}" -C "${destdir}" && rm -f "${archive}"
}

# ---------------------------------------------------------------------------
# Train split  (token m1639953) — S2 + S1, all regions
# ---------------------------------------------------------------------------
TRAIN_TOKEN="m1639953"
TRAIN_DIR="${DEST}/train"
mkdir -p "${TRAIN_DIR}"

TRAIN_FILES=(
    s2_africa.tar.gz
    s2_america.tar.gz
    s2_asiaEast.tar.gz
    s2_asiaWest.tar.gz
    s2_europa.tar.gz
    s1_africa.tar.gz
    s1_america.tar.gz
    s1_asiaEast.tar.gz
    s1_asiaWest.tar.gz
    s1_europa.tar.gz
)

for f in "${TRAIN_FILES[@]}"; do
    region="${f%.tar.gz}"           # e.g. s2_africa
    extracted="${TRAIN_DIR}/${region}"
    if [[ -d "${extracted}" ]]; then
        echo "[SKIP]     ${f} (already extracted)"
        continue
    fi
    _dl "${TRAIN_TOKEN}" "${f}" "${DEST}/${f}"
    _extract "${DEST}/${f}" "${TRAIN_DIR}"
done

# ---------------------------------------------------------------------------
# Test split  (token m1659251) — S2 + S1, all regions
# ---------------------------------------------------------------------------
TEST_TOKEN="m1659251"
TEST_DIR="${DEST}/test"
mkdir -p "${TEST_DIR}"

TEST_FILES=(
    s2_africa_test.tar.gz
    s2_america_test.tar.gz
    s2_asiaEast_test.tar.gz
    s2_asiaWest_test.tar.gz
    s2_europa_test.tar.gz
    s1_africa_test.tar.gz
    s1_america_test.tar.gz
    s1_asiaEast_test.tar.gz
    s1_asiaWest_test.tar.gz
    s1_europa_test.tar.gz
)

for f in "${TEST_FILES[@]}"; do
    region="${f%_test.tar.gz}"
    extracted="${TEST_DIR}/${region}_test"
    if [[ -d "${extracted}" ]]; then
        echo "[SKIP]     ${f} (already extracted)"
        continue
    fi
    _dl "${TEST_TOKEN}" "${f}" "${DEST}/${f}"
    _extract "${DEST}/${f}" "${TEST_DIR}"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SEN12MS-CR-TS ready at: ${DEST}"
echo "  Done: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
