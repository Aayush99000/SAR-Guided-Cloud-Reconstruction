#!/usr/bin/env python
"""Fetch ESA WorldCover labels for every SEN12MS-CR-TS patch.

For each tile, one windowed read of the WorldCover COG(s) covering the tile
(over HTTPS, no full download), then each patch is resampled onto its own
Sentinel-2 grid with mode resampling and mapped to class indices.

Output: <out>/<train|test>/<region>__<roi>__<tile>_wc.npy, (P, H, W) uint8,
rows aligned with the cloud-mask index.  Resumable: existing tiles are skipped.

    python scripts/fetch_worldcover.py --config configs/sen12mscrts.yaml --workers 8
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import OmegaConf
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject, transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.sen12mscrts_dataset import SEN12MSCRTSDataset  # noqa: E402
from data.worldcover import wc_label_path, wc_to_index  # noqa: E402

log = logging.getLogger("fetch_worldcover")

_URLS = {
    2020: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{}_Map.tif",
    2021: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif",
}


def _wc_tile_names(west: float, south: float, east: float, north: float):
    """WorldCover 3x3-degree tiles (named by lower-left corner) intersecting the bounds."""
    for lat in range(math.floor(south / 3) * 3, math.floor(north / 3) * 3 + 1, 3):
        for lon in range(math.floor(west / 3) * 3, math.floor(east / 3) * 3 + 1, 3):
            yield f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def _patch_grid(base: SEN12MSCRTSDataset, tile: dict, row: int):
    """Transform and CRS of a patch, from its first available S2 date."""
    patch = int(tile["patch_ids"][row])
    for t in np.flatnonzero(tile["valid"][row]):
        path = base._s2_path(tile, int(t), patch)
        if path.exists():
            with rasterio.open(path) as src:
                return src.transform, src.crs
    return None, None


def fetch_tile(base: SEN12MSCRTSDataset, t_idx: int, out_path: Path, year: int) -> str:
    tile = base._tiles[t_idx]
    h, w = tile["hw"]
    grids = [_patch_grid(base, tile, r) for r in range(len(tile["patch_ids"]))]
    known = [g for g in grids if g[0] is not None]
    if not known:
        raise RuntimeError(f"{tile['key']}: no readable S2 patch")

    # Union of patch bounds in lon/lat (patches may be in UTM), plus a margin
    # of a few WorldCover pixels.
    corners = []
    for tf, crs in known:
        xs, ys = sorted([tf.c, tf.c + tf.a * w]), sorted([tf.f, tf.f + tf.e * h])
        corners.append(transform_bounds(crs, "EPSG:4326", xs[0], ys[0], xs[1], ys[1]))
    corners = np.array(corners)
    pad = 5 / 12000
    bounds = (corners[:, 0].min() - pad, corners[:, 1].min() - pad,
              corners[:, 2].max() + pad, corners[:, 3].max() + pad)

    sources = []
    for name in _wc_tile_names(*bounds):
        try:
            sources.append(rasterio.open("/vsicurl/" + _URLS[year].format(name)))
        except rasterio.errors.RasterioIOError:
            pass  # no WorldCover tile there (open ocean)
    labels = np.full((len(grids), h, w), 255, dtype=np.uint8)
    if sources:
        mosaic, mosaic_tf = merge(sources, bounds=bounds, nodata=0)
        for src in sources:
            src.close()
        for r, (tf, crs) in enumerate(grids):
            if tf is None:
                continue
            dst = np.zeros((h, w), dtype=np.uint8)
            reproject(mosaic[0], dst, src_transform=mosaic_tf, src_crs="EPSG:4326", dst_transform=tf,
                      dst_crs=crs, src_nodata=0, dst_nodata=0, resampling=Resampling.mode)
            labels[r] = wc_to_index(dst)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npy")
    np.save(tmp, labels)
    os.replace(tmp, out_path)
    valid = float((labels != 255).mean())
    return f"{tile['key']}: {len(grids)} patches, {len(sources)} WC tile(s), {valid:.1%} labelled"


_BASES = {}


def _init_worker(root, mask_dir):
    logging.getLogger("data.sen12mscrts_dataset").setLevel(logging.WARNING)
    for split in ("train", "test"):
        _BASES[split] = SEN12MSCRTSDataset(root, split=split, mask_dir=mask_dir, val_tiles=[])


def _run(split, t_idx, out_path, year):
    return fetch_tile(_BASES[split], t_idx, out_path, year)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True, help="YAML(s) merged on top of default.yaml")
    ap.add_argument("--out", default=None, help="Output root (default <data_root>/worldcover)")
    ap.add_argument("--year", type=int, default=2020, choices=sorted(_URLS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-tiles", type=int, default=0, help="Only the first N tiles per split (smoke test)")
    args, overrides = ap.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    root = Path(__file__).resolve().parent.parent / "configs"
    cfg = OmegaConf.load(root / "default.yaml")
    if (root / "paths.yaml").exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "paths.yaml"))
    # --config is nargs="+", so it swallows dotlist overrides that follow it
    overrides = [c for c in args.config if "=" in c] + overrides
    for extra in (c for c in args.config if "=" not in c):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    data_root = Path(cfg.paths.data_root)
    mask_dir = cfg.paths.get("ts_mask_dir", None)
    out_root = Path(args.out) if args.out else data_root / "worldcover"

    jobs = []
    for split in ("train", "test"):
        base = SEN12MSCRTSDataset(data_root, split=split, mask_dir=mask_dir, val_tiles=[])
        tiles = range(len(base._tiles))
        if args.limit_tiles:
            tiles = tiles[: args.limit_tiles]
        for t_idx in tiles:
            out_path = wc_label_path(out_root, split, base._tiles[t_idx])
            if not out_path.exists():
                jobs.append((split, t_idx, out_path))
    log.info("%d tiles to fetch (WorldCover %d) -> %s", len(jobs), args.year, out_root)

    t0, failed = time.time(), 0
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(data_root, mask_dir)) as pool:
        futures = {pool.submit(_run, s, t, p, args.year): p for s, t, p in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                log.info("[%d/%d %.0fs] %s", i, len(jobs), time.time() - t0, fut.result())
            except Exception as exc:  # keep going; rerun to retry
                failed += 1
                log.error("[%d/%d] %s failed: %s", i, len(jobs), futures[fut].name, exc)
    if failed:
        raise SystemExit(f"{failed} tile(s) failed; rerun to retry")
    log.info("Done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
