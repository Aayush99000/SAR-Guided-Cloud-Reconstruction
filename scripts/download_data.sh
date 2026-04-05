#!/bin/bash
# =============================================================================
# Download, extract, and reorganise SEN12MS-CR (and optionally SEN12MS-CR-TS).
#
# Official dataset source:  https://patricktum.github.io/cloud_removal/
# Paper: Ebel et al. (2022) "SEN12MS-CR-TS: A Remote Sensing Data Set for
#        Multi-modal Multi-temporal Cloud Removal"
#
# Usage:
#   chmod +x scripts/download_data.sh
#   ./scripts/download_data.sh
#
# After running, the SEN12MS-CR directory will be reorganised into the flat
# layout expected by data/sen12mscr_dataset.py:
#
#   <dest>/SEN12MSCR/
#       s1/                        <- all S1 .tif patches (flat)
#       s2/                        <- all S2 cloud-free .tif patches (flat)
#       s2_cloudy/                 <- all S2 cloudy .tif patches (flat)
#       splits/
#           train.csv              <- 70 % split
#           val.csv                <- 15 % split
#           test.csv               <- 15 % split
#
# Each CSV has columns: patch_id, s1, s2_clean, s2_cloudy, season, roi
# =============================================================================

set -euo pipefail

clear
echo "SEN12MS-CR / SEN12MS-CR-TS download script"
echo "See: https://patricktum.github.io/cloud_removal/"
echo

# ---------------------------------------------------------------------------
# User prompts (identical to the official PatrickTUM script)
# ---------------------------------------------------------------------------

while true; do
    read -p "Download the multitemporal SEN12MS-CR-TS data set? [y/n] " yn
    case $yn in
        [Yy]* ) SEN12MSCRTS=true;  break;;
        [Nn]* ) SEN12MSCRTS=false; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

if [ "$SEN12MSCRTS" = "true" ]; then
    while true; do
        read -p "Regions? [all|africa|america|asiaEast|asiaWest|europa] " region
        case $region in
            all|africa|america|asiaEast|asiaWest|europa ) reg=$region; break;;
            * ) echo "Please answer [all|africa|america|asiaEast|asiaWest|europa].";;
        esac
    done
fi

while true; do
    read -p "Also download the monotemporal SEN12MS-CR data set (all regions)? [y/n] " yn
    case $yn in
        [Yy]* ) SEN12MSCR=true;  break;;
        [Nn]* ) SEN12MSCR=false; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

while true; do
    read -p "Also download the Sentinel-1 SAR data for your choices? [y/n] " yn
    case $yn in
        [Yy]* ) S1=true;  break;;
        [Nn]* ) S1=false; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo
read -p "Path to download and extract data to: " dl_extract_to
echo

# ---------------------------------------------------------------------------
# URL + size registry  (verbatim from the official PatrickTUM dl_data.sh)
# ---------------------------------------------------------------------------

declare -A url_dict
declare -A vol_dict

# ---- SEN12MS-CR-TS (multitemporal) ----------------------------------------
if [ "$SEN12MSCRTS" = "true" ]; then
    echo "Queuing SEN12MS-CR-TS download..."
    mkdir -p "$dl_extract_to/SEN12MSCRTS"

    case $region in
        all|africa)
            url_dict['multi_s2_africa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_africa.tar.gz'
            vol_dict['multi_s2_africa']='98233900'
            ;;
    esac
    case $region in
        all|america)
            url_dict['multi_s2_america']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_america.tar.gz'
            vol_dict['multi_s2_america']='110245004'
            ;;
    esac
    case $region in
        all|asiaEast)
            url_dict['multi_s2_asiaEast']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_asiaEast.tar.gz'
            vol_dict['multi_s2_asiaEast']='113948560'
            ;;
    esac
    case $region in
        all|asiaWest)
            url_dict['multi_s2_asiaWest']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_asiaWest.tar.gz'
            vol_dict['multi_s2_asiaWest']='96082796'
            ;;
    esac
    case $region in
        all|europa)
            url_dict['multi_s2_europa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s2_europa.tar.gz'
            vol_dict['multi_s2_europa']='196669740'
            ;;
    esac

    # test splits
    case $region in
        all|africa)
            url_dict['multi_s2_africa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_africa_test.tar.gz'
            vol_dict['multi_s2_africa_test']='25421744'
            ;;
    esac
    case $region in
        all|america)
            url_dict['multi_s2_america_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_america_test.tar.gz'
            vol_dict['multi_s2_america_test']='25421824'
            ;;
    esac
    case $region in
        all|asiaEast)
            url_dict['multi_s2_asiaEast_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_asiaEast_test.tar.gz'
            vol_dict['multi_s2_asiaEast_test']='40534760'
            ;;
    esac
    case $region in
        all|asiaWest)
            url_dict['multi_s2_asiaWest_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_asiaWest_test.tar.gz'
            vol_dict['multi_s2_asiaWest_test']='15012924'
            ;;
    esac
    case $region in
        all|europa)
            url_dict['multi_s2_europa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s2_europa_test.tar.gz'
            vol_dict['multi_s2_europa_test']='79568460'
            ;;
    esac

    if [ "$S1" = "true" ]; then
        case $region in
            all|africa)
                url_dict['multi_s1_africa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_africa.tar.gz'
                vol_dict['multi_s1_africa']='60544524'
                url_dict['multi_s1_africa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_africa_test.tar.gz'
                vol_dict['multi_s1_africa_test']='15668120'
                ;;
        esac
        case $region in
            all|america)
                url_dict['multi_s1_america']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_america.tar.gz'
                vol_dict['multi_s1_america']='67947416'
                url_dict['multi_s1_america_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_america_test.tar.gz'
                vol_dict['multi_s1_america_test']='15668160'
                ;;
        esac
        case $region in
            all|asiaEast)
                url_dict['multi_s1_asiaEast']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_asiaEast.tar.gz'
                vol_dict['multi_s1_asiaEast']='70230104'
                url_dict['multi_s1_asiaEast_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_asiaEast_test.tar.gz'
                vol_dict['multi_s1_asiaEast_test']='24982736'
                ;;
        esac
        case $region in
            all|asiaWest)
                url_dict['multi_s1_asiaWest']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_asiaWest.tar.gz'
                vol_dict['multi_s1_asiaWest']='59218848'
                url_dict['multi_s1_asiaWest_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_asiaWest_test.tar.gz'
                vol_dict['multi_s1_asiaWest_test']='9252904'
                ;;
        esac
        case $region in
            all|europa)
                url_dict['multi_s1_europa']='https://dataserv.ub.tum.de/s/m1639953/download?path=/&files=s1_europa.tar.gz'
                vol_dict['multi_s1_europa']='121213836'
                url_dict['multi_s1_europa_test']='https://dataserv.ub.tum.de/s/m1659251/download?path=/&files=s1_europa_test.tar.gz'
                vol_dict['multi_s1_europa_test']='49040432'
                ;;
        esac
    fi
fi

# ---- SEN12MS-CR (monotemporal, all regions) --------------------------------
if [ "$SEN12MSCR" = "true" ]; then
    echo "Queuing SEN12MS-CR download..."
    mkdir -p "$dl_extract_to/SEN12MSCR"

    url_dict['mono_s2_spring']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1158_spring_s2.tar.gz'
    vol_dict['mono_s2_spring']='48568904'
    url_dict['mono_s2_summer']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1868_summer_s2.tar.gz'
    vol_dict['mono_s2_summer']='56425520'
    url_dict['mono_s2_fall']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1970_fall_s2.tar.gz'
    vol_dict['mono_s2_fall']='68291864'
    url_dict['mono_s2_winter']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs2017_winter_s2.tar.gz'
    vol_dict['mono_s2_winter']='30580552'

    url_dict['mono_s2_cloudy_spring']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1158_spring_s2_cloudy.tar.gz'
    vol_dict['mono_s2_cloudy_spring']='48569368'
    url_dict['mono_s2_cloudy_summer']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1868_summer_s2_cloudy.tar.gz'
    vol_dict['mono_s2_cloudy_summer']='56426004'
    url_dict['mono_s2_cloudy_fall']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1970_fall_s2_cloudy.tar.gz'
    vol_dict['mono_s2_cloudy_fall']='68292448'
    url_dict['mono_s2_cloudy_winter']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs2017_winter_s2_cloudy.tar.gz'
    vol_dict['mono_s2_cloudy_winter']='30580812'

    if [ "$S1" = "true" ]; then
        url_dict['mono_s1_spring']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1158_spring_s1.tar.gz'
        vol_dict['mono_s1_spring']='15026120'
        url_dict['mono_s1_summer']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1868_summer_s1.tar.gz'
        vol_dict['mono_s1_summer']='17456784'
        url_dict['mono_s1_fall']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs1970_fall_s1.tar.gz'
        vol_dict['mono_s1_fall']='21127832'
        url_dict['mono_s1_winter']='https://dataserv.ub.tum.de/s/m1554803/download?path=/&files=ROIs2017_winter_s1.tar.gz'
        vol_dict['mono_s1_winter']='9460956'
    fi
fi

# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------

req=0
for key in "${!vol_dict[@]}"; do
    req=$((req + ${vol_dict[$key]}))
done

avail=$(df "$dl_extract_to" | awk 'NR==2 { print $4 }')
echo
if (( avail < req )); then
    echo "ERROR: Not enough disk space on $dl_extract_to"
    echo "  Available : $avail (512-byte sectors)"
    echo "  Required  : $req (512-byte sectors)"
    exit 1
else
    echo "Disk check OK — using $req of $avail available 512-byte sectors on $dl_extract_to"
fi
echo

# ---------------------------------------------------------------------------
# Download + extract
#
# TUM dataserv is a NextCloud instance.  The /s/<id>/download path returns a
# 303 → WebDAV redirect that wget mishandles (gets 0 bytes).  The correct
# stable URL is /index.php/s/<id>/download — curl follows this cleanly.
# ---------------------------------------------------------------------------

_download_file() {
    local url="$1"
    local dest="$2"

    # Rewrite:  /s/<id>/download  →  /index.php/s/<id>/download
    local fixed_url
    fixed_url=$(echo "$url" | sed 's|/s/\(m[0-9]*\)/download|/index.php/s/\1/download|')

    echo "[DOWNLOAD] $(basename "$dest")"
    echo "           $fixed_url"

    # -L  follow redirects
    # -k  skip SSL verification (mirrors wget --no-check-certificate)
    # -C- resume partial downloads
    # --retry 5 with exponential backoff
    curl -L -k -C - \
         --retry 5 --retry-delay 5 --retry-max-time 120 \
         --progress-bar \
         -o "$dest" \
         "$fixed_url"

    # Sanity-check: fail loudly on empty file rather than a cryptic tar error
    local size
    size=$(wc -c < "$dest")
    if (( size < 1024 )); then
        echo "ERROR: Downloaded file is only ${size} bytes — server likely returned" \
             "an error page.  Check the URL or try again later."
        rm -f "$dest"
        exit 1
    fi
}

for key in "${!url_dict[@]}"; do
    url="${url_dict[$key]}"
    filename="${url##*files=}"
    dest="$dl_extract_to/$filename"

    _download_file "$url" "$dest"

    echo "[EXTRACT]  $filename"
    tar --extract --file "$dest" -C "$dl_extract_to"
    rm "$dest"
done

# ---------------------------------------------------------------------------
# Move extracted dirs into SEN12MSCR / SEN12MSCRTS  (official logic)
# ---------------------------------------------------------------------------

echo "Moving data into place (please don't interrupt)..."
for key in "${!url_dict[@]}"; do
    url="${url_dict[$key]}"
    filename="${url##*files=}"
    dirname="${filename%.tar.gz}"

    if [[ "$url" == *"m1554803"* ]]; then
        # SEN12MS-CR — keep the per-season subdir inside SEN12MSCR/
        mv "$dl_extract_to/$dirname" "$dl_extract_to/SEN12MSCR/" 2>/dev/null || true
    elif [[ "$url" == *"m1639953"* ]]; then
        # SEN12MS-CR-TS train — strip the leading "s2_" / "s1_" prefix
        no_prefix="${dirname#s2_}"
        no_prefix="${no_prefix#s1_}"
        rsync -a --remove-source-files \
            "$dl_extract_to/$no_prefix/" \
            "$dl_extract_to/SEN12MSCRTS/" 2>/dev/null || true
        rm -rf "$dl_extract_to/$no_prefix"
    else
        # SEN12MS-CR-TS test
        rsync -a --remove-source-files \
            "$dl_extract_to/$dirname/" \
            "$dl_extract_to/SEN12MSCRTS/" 2>/dev/null || true
        rm -rf "$dl_extract_to/$dirname"
    fi
done

echo "Download and extraction complete."
echo

# ---------------------------------------------------------------------------
# Reorganise SEN12MS-CR into the flat layout expected by SEN12MSCRDataset
#
# Input (per-season subdirs, original layout):
#   SEN12MSCR/
#       ROIs1158_spring_s1/       ROIs1158_spring_s1_p1.tif ...
#       ROIs1158_spring_s2/       ROIs1158_spring_s2_p1.tif ...
#       ROIs1158_spring_s2_cloudy/ ROIs1158_spring_s2_cloudy_p1.tif ...
#       ROIs1868_summer_s1/ ...
#       ...
#
# Output (flat dirs + CSV splits):
#   SEN12MSCR/
#       s1/            all *_s1_p*.tif files
#       s2/            all *_s2_p*.tif files   (cloud-free)
#       s2_cloudy/     all *_s2_cloudy_p*.tif files
#       splits/
#           train.csv  70 %
#           val.csv    15 %
#           test.csv   15 %
# ---------------------------------------------------------------------------

if [ "$SEN12MSCR" = "true" ]; then
    echo "Reorganising SEN12MS-CR into flat layout..."
    MSCR="$dl_extract_to/SEN12MSCR"

    mkdir -p "$MSCR/s1" "$MSCR/s2" "$MSCR/s2_cloudy" "$MSCR/splits"

    # Move files from per-season subdirs into flat target dirs
    # S2 cloud-free
    find "$MSCR" -maxdepth 2 -path "*/ROIs*_s2/*_s2_p*.tif" \
        ! -path "*cloudy*" -exec mv {} "$MSCR/s2/" \;

    # S2 cloudy
    find "$MSCR" -maxdepth 2 -path "*/ROIs*_s2_cloudy/*.tif" \
        -exec mv {} "$MSCR/s2_cloudy/" \;

    # S1 (only if downloaded)
    if [ "$S1" = "true" ]; then
        find "$MSCR" -maxdepth 2 -path "*/ROIs*_s1/*.tif" \
            -exec mv {} "$MSCR/s1/" \;
    fi

    # Remove now-empty per-season subdirs
    find "$MSCR" -maxdepth 1 -type d -name "ROIs*" -exec rm -rf {} + 2>/dev/null || true

    echo "Flat layout created. Generating CSV splits..."

    # Generate splits with Python (available in any standard conda/venv)
    python3 - "$MSCR" "$S1" <<'PYEOF'
import sys, csv, random, pathlib

mscr    = pathlib.Path(sys.argv[1])
has_s1  = sys.argv[2].lower() == "true"

s2_dir       = mscr / "s2"
s2c_dir      = mscr / "s2_cloudy"
s1_dir       = mscr / "s1"
splits_dir   = mscr / "splits"

# ---------------------------------------------------------------------------
# Build complete triplets (patches where ALL three modalities exist)
# ---------------------------------------------------------------------------
records = []
for s2_path in sorted(s2_dir.glob("*.tif")):
    stem = s2_path.stem                     # e.g. ROIs1158_spring_s2_p42
    # Derive sibling filenames
    s2c_name = stem.replace("_s2_p", "_s2_cloudy_p") + ".tif"
    s1_name  = stem.replace("_s2_p", "_s1_p") + ".tif"

    s2c_path = s2c_dir / s2c_name
    s1_path  = s1_dir  / s1_name

    if not s2c_path.exists():
        continue
    if has_s1 and not s1_path.exists():
        continue

    # Parse season and ROI from filename  (ROIs1158_spring_s2_p42)
    parts  = stem.split("_")           # ['ROIs1158', 'spring', 's2', 'p42']
    roi    = parts[0]                  # ROIs1158
    season = parts[1]                  # spring
    patch  = parts[-1]                 # p42
    pid    = f"{roi}_{season}_{patch}" # ROIs1158_spring_p42

    records.append({
        "patch_id":  pid,
        "s1":        f"s1/{s1_name}"       if has_s1 else "",
        "s2_clean":  f"s2/{s2_path.name}",
        "s2_cloudy": f"s2_cloudy/{s2c_name}",
        "season":    season,
        "roi":       roi,
    })

if not records:
    print("WARNING: No complete triplets found — check that s2/ and s2_cloudy/ are populated.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Deterministic 70 / 15 / 15 split
# ---------------------------------------------------------------------------
random.seed(42)
random.shuffle(records)
n       = len(records)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

splits = {
    "train": records[:n_train],
    "val":   records[n_train : n_train + n_val],
    "test":  records[n_train + n_val :],
}

fieldnames = ["patch_id", "s1", "s2_clean", "s2_cloudy", "season", "roi"]
for name, rows in splits.items():
    out = splits_dir / f"{name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  {name:5s}: {len(rows):5d} patches  →  {out}")

print(f"\nTotal: {n} complete triplets across train/val/test.")
PYEOF

    echo
    echo "SEN12MS-CR is ready at: $MSCR"
    echo "  s1/          SAR patches"
    echo "  s2/          cloud-free optical patches"
    echo "  s2_cloudy/   cloudy optical patches"
    echo "  splits/      train.csv  val.csv  test.csv"
fi

echo
echo "All done! Enjoy :)"
