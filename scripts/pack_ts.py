"""
Pack SEN12MS-CR-TS tiles into one memory-mappable array per tile and modality.

The TUM GeoTIFFs are pixel-interleaved, so reading 4 of the 13 S2 bands still
reads the whole 1.7 MB file, and a training sample opens 7 files.  On /scratch
the data loader then spends most of its time waiting on I/O.  This writes

    <pack_dir>/<split>/<region>__<roi>__<tile>_s2.npy   (P, D, B, H, W) uint16  raw S2 bands
    <pack_dir>/<split>/<region>__<roi>__<tile>_s1.npy   (P, D, 2, H, W) float32 raw S1 dB

with rows in the order of the cloud-mask index (``patch_ids``), D dates and B =
the configured optical bands.  Values are copied unchanged, so samples read from
the pack are bit-identical to samples read from the GeoTIFFs.  Dates without a
valid file stay zero (the loader never selects them).  Each tile is written
under a ``.partial`` name and renamed once complete, so a loader never sees a
half-written tile.

    python scripts/pack_ts.py --config configs/default.yaml configs/sen12mscrts.yaml \
        --out Dataset/SEN12MSCRTS/packed --split train --workers 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sen12mscrts_dataset import OFFICIAL_TEST_TILES  # noqa: E402


def _dirs(root: Path, split: str, region: str, roi: str, tile: str):
    if split == "test":
        return (root / "test" / f"s2_{region}_test" / roi / tile / "S2",
                root / "test" / f"s1_{region}_test" / roi / tile / "S1")
    return root / "train" / region / roi / tile / "S2", root / "train" / region / roi / tile / "S1"


def _pack_date(job):
    """Copy every patch of one date into the tile's partial arrays."""
    s2_out, s1_out, s2_dir, s1_dir, roi, tile, t, s2_date, s1_date, patch_ids, s2_ok, s1_ok, bands = job
    s2 = np.load(s2_out, mmap_mode="r+")
    s1 = np.load(s1_out, mmap_mode="r+")
    for row, patch in enumerate(patch_ids):
        if s2_ok[row]:
            with rasterio.open(s2_dir / str(t) / f"s2_{roi}_{tile}_ImgNo_{t}_{s2_date}_patch_{patch}.tif") as src:
                s2[row, t] = src.read([b + 1 for b in bands])
        if s1_ok[row]:
            with rasterio.open(s1_dir / str(t) / f"s1_{roi}_{tile}_ImgNo_{t}_{s1_date}_patch_{patch}.tif") as src:
                s1[row, t] = src.read()
    s2.flush()
    s1.flush()
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="pack directory (paths.ts_pack_dir)")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--include-test-copies", action="store_true",
                    help="also pack the copies of official test tiles under train/ (legacy split only)")
    ap.add_argument("--only", nargs="*", help="pack only these region/roi/tile keys")
    args = ap.parse_args()

    cfg = OmegaConf.merge(*[OmegaConf.load(c) for c in args.config])
    root = Path(cfg.paths.data_root)
    mask_dir = Path(cfg.paths.get("ts_mask_dir") or root / "cloud_masks") / args.split
    bands = [int(b) for b in cfg.data.optical_bands]
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "bands.json").write_text(json.dumps({"optical_bands": bands}))

    metas = sorted(mask_dir.glob("*_meta.npz"))
    with Pool(args.workers) as pool:
        for mf in metas:
            name = mf.name[: -len("_meta.npz")]
            region, roi, tile = name.split("__")
            if args.only and f"{region}/{roi}/{tile}" not in args.only:
                continue
            if args.split == "train" and not args.include_test_copies and f"{region}/{roi}/{tile}" in OFFICIAL_TEST_TILES:
                print(f"skip {name} (copy of an official test tile)", flush=True)
                continue
            s2_final, s1_final = out_dir / f"{name}_s2.npy", out_dir / f"{name}_s1.npy"
            if s2_final.exists() and s1_final.exists():
                print(f"done {name}", flush=True)
                continue
            with np.load(mf, allow_pickle=False) as z:
                meta = {k: z[k] for k in z.files}
            n_p, n_d = meta["s2_ok"].shape
            h, w = int(meta["height"]), int(meta["width"])
            s2_part, s1_part = out_dir / f"{name}_s2.partial.npy", out_dir / f"{name}_s1.partial.npy"
            np.lib.format.open_memmap(s2_part, mode="w+", dtype=np.uint16, shape=(n_p, n_d, len(bands), h, w)).flush()
            np.lib.format.open_memmap(s1_part, mode="w+", dtype=np.float32, shape=(n_p, n_d, 2, h, w)).flush()
            s2_dir, s1_dir = _dirs(root, args.split, region, roi, tile)
            jobs = [(s2_part, s1_part, s2_dir, s1_dir, roi, tile, t, str(meta["s2_dates"][t]), str(meta["s1_dates"][t]),
                     [int(p) for p in meta["patch_ids"]], meta["s2_ok"][:, t], meta["s1_ok"][:, t], bands)
                    for t in range(n_d)]
            t0 = time.time()
            for _ in pool.imap_unordered(_pack_date, jobs):
                pass
            s2_part.rename(s2_final)
            s1_part.rename(s1_final)
            gb = (s2_final.stat().st_size + s1_final.stat().st_size) / 1e9
            print(f"packed {name}: {n_p} patches x {n_d} dates, {gb:.1f} GB in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
