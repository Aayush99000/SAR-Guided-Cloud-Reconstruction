"""
Precompute s2cloudless cloud masks and per-date cloud coverage for SEN12MS-CR-TS.

SEN12MS-CR-TS has no labelled cloud-free target: every location has ~30 S2
acquisitions of varying cloudiness, and the loader picks the target and inputs
from per-date cloud coverage.  Running s2cloudless inside the DataLoader is far
too slow (~2.4 s per 256x256 image on one core), so this script runs it once.

Detector settings follow UnCRtainTS (Ebel et al.): s2cloudless with
threshold=0.4, average_over=4, dilation_size=2, on clip(S2, 0, 10000) / 10000.

Work is split into (tile, time-step) units so all cores stay busy, and tiles
can be sharded across a SLURM array with --shard / --num-shards.

Output, one pair of files per tile, under <out_dir>/<split>/:
    <region>__<roi>__<tile>_meta.npz
        patch_ids  (P,)    int     patch numbers
        coverage   (P, T)  float32 cloud fraction per date, NaN where S2 missing
        s2_ok      (P, T)  bool
        s1_ok      (P, T)  bool
        s2_dates   (T,)    str     acquisition date per time-step folder ('' if absent)
        s1_dates   (T,)    str
        height, width      int
    <region>__<roi>__<tile>_masks.npy
        (P, T, H*W/8) uint8, np.packbits of the binary cloud mask (1 = cloud);
        uncompressed so the loader can memory-map it.

Usage:
    python scripts/precompute_ts_clouds.py \\
        --root /scratch/.../Dataset/SEN12MSCRTS \\
        --out-dir /scratch/.../Dataset/SEN12MSCRTS/cloud_masks \\
        --workers 32 [--shard 0 --num-shards 4] [--limit-tiles 1]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

_FNAME_RE = re.compile(r"ImgNo_(\d+)_(\d{4}-\d{2}-\d{2})_patch_(\d+)\.tif$")

_DETECTOR = None


def iter_tiles(root: Path) -> Iterator[Dict]:
    """Yield one dict per tile for both the train and test trees."""
    train = root / "train"
    for region_dir in sorted(p for p in train.iterdir() if p.is_dir()):
        for roi_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            for tile_dir in sorted(p for p in roi_dir.iterdir() if p.is_dir()):
                yield dict(
                    split="train", region=region_dir.name, roi=roi_dir.name,
                    tile=tile_dir.name, s2_dir=tile_dir / "S2", s1_dir=tile_dir / "S1",
                )

    test = root / "test"
    for s2_region in sorted(test.glob("s2_*_test")):
        region = s2_region.name[len("s2_"):-len("_test")]
        s1_region = test / f"s1_{region}_test"
        for roi_dir in sorted(p for p in s2_region.iterdir() if p.is_dir()):
            for tile_dir in sorted(p for p in roi_dir.iterdir() if p.is_dir()):
                yield dict(
                    split="test", region=region, roi=roi_dir.name, tile=tile_dir.name,
                    s2_dir=tile_dir / "S2",
                    s1_dir=s1_region / roi_dir.name / tile_dir.name / "S1",
                )


def _scan_modality(mod_dir: Path) -> Dict[int, Tuple[str, Dict[int, Path]]]:
    """Return {t: (date, {patch: path})} for one modality of one tile."""
    out = {}
    if not mod_dir.is_dir():
        return out
    for t_dir in mod_dir.iterdir():
        if not t_dir.is_dir() or not t_dir.name.isdigit():
            continue
        dates, patches = set(), {}
        for f in t_dir.iterdir():
            m = _FNAME_RE.search(f.name)
            if m is None:
                continue
            dates.add(m.group(2))
            patches[int(m.group(3))] = f
        if len(dates) > 1:
            raise RuntimeError(f"Multiple dates in one time-step folder: {t_dir} -> {sorted(dates)}")
        out[int(t_dir.name)] = (dates.pop() if dates else "", patches)
    return out


def _init_worker() -> None:
    # One lightgbm thread per process; parallelism comes from the process pool.
    os.environ["OMP_NUM_THREADS"] = "1"
    global _DETECTOR
    from s2cloudless import S2PixelCloudDetector
    _DETECTOR = S2PixelCloudDetector(
        threshold=0.4, average_over=4, dilation_size=2, all_bands=True,
    )


def process_unit(items: List[Tuple[int, str]], h: int, w: int, batch_size: int):
    """Run s2cloudless over one (tile, time-step). items = [(row, path), ...]."""
    import rasterio

    rows_out, cov_out, masks_out, warnings = [], [], [], []
    for b in range(0, len(items), batch_size):
        arrs, rows = [], []
        for row, path in items[b:b + batch_size]:
            try:
                with rasterio.open(path) as src:
                    a = src.read().astype(np.float32)
            except Exception as exc:
                warnings.append(f"unreadable {path}: {exc}")
                continue
            if a.shape != (13, h, w):
                warnings.append(f"unexpected shape {a.shape} in {path}")
                continue
            arrs.append(np.clip(np.nan_to_num(a), 0, 10000) / 10000.0)
            rows.append(row)
        if not arrs:
            continue
        cm = _DETECTOR.get_cloud_masks(np.stack(arrs).transpose(0, 2, 3, 1)).astype(bool)
        for r, m in zip(rows, cm):
            rows_out.append(r)
            cov_out.append(m.mean())
            masks_out.append(np.packbits(m.reshape(-1)))
    return rows_out, cov_out, masks_out, warnings


def _prepare_tile(tile: Dict, out_dir: Path):
    """Scan a tile and allocate its output arrays. Returns None if already done."""
    import rasterio

    stem = f"{tile['region']}__{tile['roi']}__{tile['tile']}"
    split_dir = out_dir / tile["split"]
    meta_path = split_dir / f"{stem}_meta.npz"
    mask_path = split_dir / f"{stem}_masks.npy"
    if meta_path.exists() and mask_path.exists():
        return None

    s2 = _scan_modality(tile["s2_dir"])
    s1 = _scan_modality(tile["s1_dir"])
    if not s2:
        raise RuntimeError(f"no S2 data found in {tile['s2_dir']}")

    n_t = max(max(s2), max(s1) if s1 else -1) + 1
    patch_ids = sorted(set().union(*(p for _, p in s2.values())))
    p_index = {p: i for i, p in enumerate(patch_ids)}
    n_p = len(patch_ids)

    first_file = next(f for _, p in s2.values() for f in p.values())
    with rasterio.open(first_file) as src:
        h, w = src.height, src.width

    s1_ok = np.zeros((n_p, n_t), dtype=bool)
    for t, (_, patches) in s1.items():
        for p in patches:
            if p in p_index:
                s1_ok[p_index[p], t] = True

    units = {
        t: [(p_index[p], str(path)) for p, path in sorted(patches.items())]
        for t, (_, patches) in sorted(s2.items()) if patches
    }
    state = dict(
        name=f"{tile['split']}/{stem}", meta_path=meta_path, mask_path=mask_path, h=h, w=w,
        patch_ids=np.array(patch_ids), s1_ok=s1_ok,
        coverage=np.full((n_p, n_t), np.nan, dtype=np.float32),
        s2_ok=np.zeros((n_p, n_t), dtype=bool),
        masks=np.zeros((n_p, n_t, (h * w + 7) // 8), dtype=np.uint8),
        s2_dates=np.array([s2.get(t, ("", {}))[0] for t in range(n_t)]),
        s1_dates=np.array([s1.get(t, ("", {}))[0] for t in range(n_t)]),
        pending=len(units), t0=time.time(),
    )
    return state, units


def _save_tile(st: Dict) -> str:
    st["meta_path"].parent.mkdir(parents=True, exist_ok=True)
    tmp_mask = st["mask_path"].with_name(st["mask_path"].stem + ".tmp.npy")
    tmp_meta = st["meta_path"].with_name(st["meta_path"].stem + ".tmp.npz")
    np.save(tmp_mask, st["masks"])
    np.savez(
        tmp_meta, patch_ids=st["patch_ids"], coverage=st["coverage"], s2_ok=st["s2_ok"],
        s1_ok=st["s1_ok"], s2_dates=st["s2_dates"], s1_dates=st["s1_dates"],
        height=st["h"], width=st["w"],
    )
    os.replace(tmp_mask, st["mask_path"])
    os.replace(tmp_meta, st["meta_path"])
    cov, valid = st["coverage"], st["s2_ok"] & st["s1_ok"]
    return (
        f"[DONE] {st['name']}: P={cov.shape[0]} T={cov.shape[1]} valid={valid.mean():.3f} "
        f"mean_cov={np.nanmean(cov):.3f} clear_frac={(cov < 1e-3).mean():.3f} "
        f"({(time.time() - st['t0']) / 60:.1f} min)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit-tiles", type=int, default=0, help="Process only N tiles (smoke test)")
    args = ap.parse_args()

    tiles: List[Dict] = list(iter_tiles(args.root))
    tiles = [t for i, t in enumerate(tiles) if i % args.num_shards == args.shard]
    if args.limit_tiles:
        tiles = tiles[:args.limit_tiles]

    states, jobs = {}, []
    for i, tile in enumerate(tiles):
        prepared = _prepare_tile(tile, args.out_dir)
        if prepared is None:
            print(f"[SKIP] {tile['split']}/{tile['region']}__{tile['roi']}__{tile['tile']}", flush=True)
            continue
        st, units = prepared
        states[i] = st
        jobs += [(i, t, items, st["h"], st["w"]) for t, items in units.items()]
    n_imgs = sum(len(j[2]) for j in jobs)
    print(f"[INFO] shard {args.shard}/{args.num_shards}: {len(states)} tiles to process, "
          f"{len(jobs)} units, {n_imgs} images, {args.workers} workers", flush=True)

    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(process_unit, items, h, w, args.batch_size): (i, t)
                   for i, t, items, h, w in jobs}
        for done, fut in enumerate(as_completed(futures), 1):
            i, t = futures[fut]
            st = states[i]
            try:
                rows, cov, masks, warnings = fut.result()
            except Exception as exc:
                failures += 1
                print(f"[FAIL] {st['name']} t={t}: {exc!r}", flush=True)
                st["pending"] = -1
                continue
            for msg in warnings:
                print(f"[WARN] {msg}", flush=True)
            for r, c, m in zip(rows, cov, masks):
                st["coverage"][r, t] = c
                st["s2_ok"][r, t] = True
                st["masks"][r, t] = m
            if st["pending"] > 0:
                st["pending"] -= 1
                if st["pending"] == 0:
                    print(_save_tile(st), flush=True)
                    del st["masks"]
            if done % 50 == 0:
                print(f"[PROGRESS] {done}/{len(jobs)} units", flush=True)

    if failures:
        print(f"[ERROR] {failures} unit(s) failed; affected tiles were not saved", flush=True)
        sys.exit(1)
    print("[INFO] All tiles processed.", flush=True)


if __name__ == "__main__":
    main()
