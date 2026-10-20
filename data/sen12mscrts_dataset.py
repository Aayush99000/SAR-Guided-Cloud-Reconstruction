"""
SEN12MS-CR-TS dataset (Ebel et al. 2022) for the OT-ODE diffusion bridge.

On-disk layout (as extracted from the TUM archives):

    <root>/train/<region>/<ROI>/<tile>/S1/<t>/s1_<ROI>_<tile>_ImgNo_<t>_<date>_patch_<p>.tif
    <root>/train/<region>/<ROI>/<tile>/S2/<t>/s2_<ROI>_<tile>_ImgNo_<t>_<date>_patch_<p>.tif
    <root>/test/s2_<region>_test/<ROI>/<tile>/S2/<t>/...
    <root>/test/s1_<region>_test/<ROI>/<tile>/S1/<t>/...

There is no labelled cloud-free image.  Each location has ~30 dates; per-date
s2cloudless masks are precomputed by ``scripts/precompute_ts_clouds.py`` and
this loader picks one target date and ``n_input`` input dates from them.

Selection policies
------------------
target_policy
    "least_cloudy"  argsort(coverage)[0], as in UnCRtainTS ``fixed_sampler``.
    "random_clear"  random date with coverage < clear_thresh, else least cloudy
                    (UnCRtainTS ``random_sampler``).
input_policy
    "first"    first n valid dates in time order (UnCRtainTS ``fixed_sampler``).
    "nearest"  the n valid dates closest in time to the target.
    "random"   n random valid dates, returned in time order.
exclude_target_from_inputs
    UnCRtainTS's fixed sampler does NOT exclude the target, so with
    min_cov=0/max_cov=1 the target can be one of the inputs.  Set False only
    to reproduce that protocol exactly; True avoids the model seeing the answer.
min_input_cov / max_input_cov
    Inputs are drawn only from dates whose coverage lies in this range
    (UnCRtainTS ``min_cov`` / ``max_cov``).  If a location has fewer than
    ``n_input`` such dates, all valid dates are used instead and
    ``coverage_match`` is set False.

Bridge interface
----------------
The bridge needs a single cloudy endpoint, so ``cloudy`` is a cloud-aware
composite of the inputs: each pixel comes from the least-cloudy input in which
it is clear; pixels cloudy in every input take the input mean.  ``cloud_mask``
marks those never-clear pixels.  The full stacks are returned as well
(``cloudy_stack``, ``input_masks``) for models that consume all dates.

Normalisation follows UnCRtainTS ``rescale_method='default'``:
    S1: clip [-25, 0] dB  -> [0, 1]        S2: clip [0, 10000] -> [0, 1]
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .sen12mscr_dataset import S2_RGB_NIR_BANDS

log = logging.getLogger(__name__)

_SEASON_BY_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                    6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}


def _preprocess_s1(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=-25.0, posinf=0.0, neginf=-25.0)
    return (np.clip(arr, -25.0, 0.0) + 25.0) / 25.0


def _preprocess_s2(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    return np.clip(arr, 0.0, 10000.0) / 10000.0


class SEN12MSCRTSDataset(Dataset):
    """Multi-temporal SAR-optical cloud removal on SEN12MS-CR-TS.

    Args:
        root:            Dataset root containing ``train/`` and ``test/``.
        split:           "train", "val" or "test".  "train" and "val" are
                         carved from the official train tiles by holding out
                         whole tiles; "test" is the official test split.
        mask_dir:        Output of precompute_ts_clouds.py (default
                         ``<root>/cloud_masks``).
        n_input:         Number of input dates T.
        optical_bands:   S2 band indices to return (None = all 13).
        target_policy, input_policy, exclude_target_from_inputs: see module doc.
        clear_thresh:    Coverage below which a date counts as clear.
        val_tiles:       Explicit "region/roi/tile" keys for the val split.
                         Default: the last tile (sorted) of each region.
        augment:         Random flips / 90° rotations (default: train only).
        regions:         Restrict to these regions (default: all).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        mask_dir: Optional[str | Path] = None,
        n_input: int = 3,
        optical_bands: Optional[Sequence[int]] = S2_RGB_NIR_BANDS,
        target_policy: str = "least_cloudy",
        input_policy: str = "first",
        exclude_target_from_inputs: bool = True,
        clear_thresh: float = 1e-3,
        min_input_cov: float = 0.0,
        max_input_cov: float = 1.0,
        val_tiles: Optional[Sequence[str]] = None,
        augment: Optional[bool] = None,
        regions: Optional[Sequence[str]] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if target_policy not in ("least_cloudy", "random_clear"):
            raise ValueError(f"unknown target_policy {target_policy!r}")
        if input_policy not in ("first", "nearest", "random"):
            raise ValueError(f"unknown input_policy {input_policy!r}")

        self.root = Path(root)
        self.split = split
        self.mask_dir = Path(mask_dir) if mask_dir else self.root / "cloud_masks"
        self.n_input = n_input
        self.optical_bands = list(optical_bands) if optical_bands is not None else None
        self.target_policy = target_policy
        self.input_policy = input_policy
        self.exclude_target = exclude_target_from_inputs
        self.clear_thresh = clear_thresh
        self.min_input_cov = min_input_cov
        self.max_input_cov = max_input_cov
        self.augment = (split == "train") if augment is None else augment

        src_split = "test" if split == "test" else "train"
        meta_files = sorted((self.mask_dir / src_split).glob("*_meta.npz"))
        if not meta_files:
            raise FileNotFoundError(
                f"No cloud-mask index in {self.mask_dir / src_split}. "
                "Run scripts/precompute_ts_clouds.py first."
            )

        tiles = []
        for mf in meta_files:
            region, roi, tile = mf.name[: -len("_meta.npz")].split("__")
            if regions and region not in regions:
                continue
            tiles.append((f"{region}/{roi}/{tile}", region, roi, tile, mf))

        if src_split == "train":
            if val_tiles is None:
                last_by_region: Dict[str, str] = {}
                for key, region, *_ in tiles:
                    last_by_region[region] = key
                val_set = set(last_by_region.values())
            else:
                val_set = set(val_tiles)
            keep = (lambda k: k in val_set) if split == "val" else (lambda k: k not in val_set)
            tiles = [t for t in tiles if keep(t[0])]
        self.val_tiles = sorted(val_set) if src_split == "train" else []

        self._tiles: List[Dict] = []
        self._index: List[Tuple[int, int]] = []
        n_skipped = 0
        for key, region, roi, tile, mf in tiles:
            with np.load(mf, allow_pickle=False) as z:
                meta = {k: z[k] for k in z.files}
            valid = meta["s2_ok"] & meta["s1_ok"]
            if src_split == "test":
                s2_dir = self.root / "test" / f"s2_{region}_test" / roi / tile / "S2"
                s1_dir = self.root / "test" / f"s1_{region}_test" / roi / tile / "S1"
            else:
                s2_dir = self.root / "train" / region / roi / tile / "S2"
                s1_dir = self.root / "train" / region / roi / tile / "S1"
            t_idx = len(self._tiles)
            self._tiles.append(dict(
                key=key, region=region, roi=roi, tile=tile,
                s2_dir=s2_dir, s1_dir=s1_dir,
                mask_path=mf.with_name(mf.name.replace("_meta.npz", "_masks.npy")),
                patch_ids=meta["patch_ids"], coverage=meta["coverage"], valid=valid,
                s2_dates=[str(d) for d in meta["s2_dates"]],
                s1_dates=[str(d) for d in meta["s1_dates"]],
                hw=(int(meta["height"]), int(meta["width"])),
            ))
            for row in range(len(meta["patch_ids"])):
                if valid[row].sum() >= n_input + (1 if exclude_target_from_inputs else 0):
                    self._index.append((t_idx, row))
                else:
                    n_skipped += 1

        self._masks_cache: Dict[int, np.ndarray] = {}
        log.info(
            "SEN12MSCRTSDataset | split=%s | %d tiles | %d locations (%d skipped: too few valid dates) "
            "| T=%d | target=%s inputs=%s exclude_target=%s input_cov=[%.2f, %.2f]",
            split, len(self._tiles), len(self._index), n_skipped, n_input,
            target_policy, input_policy, exclude_target_from_inputs, min_input_cov, max_input_cov,
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select(self, coverage: np.ndarray, valid: np.ndarray) -> Tuple[int, List[int], bool]:
        cov = np.where(valid, coverage, np.inf)
        valid_idx = np.flatnonzero(valid)

        if self.target_policy == "random_clear":
            clear = valid_idx[cov[valid_idx] < self.clear_thresh]
            target = int(clear[torch.randint(len(clear), (1,)).item()]) if len(clear) else int(np.argmin(cov))
        else:
            target = int(np.argsort(cov)[0])

        pool = [int(i) for i in valid_idx if not (self.exclude_target and i == target)]
        in_range = [i for i in pool if self.min_input_cov <= cov[i] <= self.max_input_cov]
        inputs_in_range = len(in_range) >= self.n_input
        if inputs_in_range:
            pool = in_range

        if self.input_policy == "first":
            inputs = pool[: self.n_input]
        elif self.input_policy == "nearest":
            inputs = sorted(sorted(pool, key=lambda i: (abs(i - target), i))[: self.n_input])
        else:
            perm = torch.randperm(len(pool))[: self.n_input].tolist()
            inputs = sorted(pool[i] for i in perm)

        coverage_match = bool(cov[target] <= self.clear_thresh) and inputs_in_range
        return target, inputs, coverage_match

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _masks(self, t_idx: int) -> np.ndarray:
        m = self._masks_cache.get(t_idx)
        if m is None:
            m = np.load(self._tiles[t_idx]["mask_path"], mmap_mode="r")
            self._masks_cache[t_idx] = m
        return m

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read()

    def _s2_path(self, tile: Dict, t: int, patch: int) -> Path:
        name = f"s2_{tile['roi']}_{tile['tile']}_ImgNo_{t}_{tile['s2_dates'][t]}_patch_{patch}.tif"
        return tile["s2_dir"] / str(t) / name

    def _s1_path(self, tile: Dict, t: int, patch: int) -> Path:
        name = f"s1_{tile['roi']}_{tile['tile']}_ImgNo_{t}_{tile['s1_dates'][t]}_patch_{patch}.tif"
        return tile["s1_dir"] / str(t) / name

    def _optical(self, arr: np.ndarray) -> np.ndarray:
        arr = _preprocess_s2(arr)
        return arr[self.optical_bands] if self.optical_bands is not None else arr

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        t_idx, row = self._index[idx]
        tile = self._tiles[t_idx]
        patch = int(tile["patch_ids"][row])
        coverage = tile["coverage"][row]
        target, inputs, coverage_match = self._select(coverage, tile["valid"][row])
        h, w = tile["hw"]

        clean = self._optical(self._read(self._s2_path(tile, target, patch)))
        cloudy_stack = np.stack([self._optical(self._read(self._s2_path(tile, t, patch))) for t in inputs])
        sar = np.concatenate([_preprocess_s1(self._read(self._s1_path(tile, t, patch))) for t in inputs])

        packed = self._masks(t_idx)[row, inputs]                              # (T, H*W/8)
        in_masks = np.unpackbits(packed, axis=-1, count=h * w).reshape(len(inputs), h, w).astype(bool)

        # Cloud-aware composite: least-cloudy clear input first.
        order = np.argsort([coverage[t] for t in inputs], kind="stable")
        composite = cloudy_stack.mean(axis=0)
        filled = np.zeros((h, w), dtype=bool)
        for k in order:
            take = ~in_masks[k] & ~filled
            composite[:, take] = cloudy_stack[k][:, take]
            filled |= take
        cloud_mask = (~filled)[None].astype(np.float32)

        tensors = {
            "sar": sar.astype(np.float32),
            "cloudy": composite.astype(np.float32),
            "cloudy_stack": cloudy_stack.astype(np.float32),
            "clean": clean.astype(np.float32),
            "cloud_mask": cloud_mask,
            "input_masks": in_masks[:, None].astype(np.float32),
        }
        if self.augment:
            k = int(torch.randint(4, (1,)))
            flip = bool(torch.randint(2, (1,)))
            for name, a in tensors.items():
                a = np.rot90(a, k, axes=(-2, -1))
                tensors[name] = np.flip(a, axis=-1) if flip else a
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in tensors.items()}

        target_date = tile["s2_dates"][target]
        out["metadata"] = {
            "patch_id": f"{tile['key']}/p{patch}",
            "cloud_coverage": float(np.mean([coverage[t] for t in inputs])),
            "composite_coverage": float(cloud_mask.mean()),
            "target_coverage": float(coverage[target]),
            "coverage_match": coverage_match,
            "target_t": target,
            "input_t": list(inputs),
            "target_date": target_date,
            "input_dates": [tile["s2_dates"][t] for t in inputs],
            "season": _SEASON_BY_MONTH[date.fromisoformat(target_date).month] if target_date else "",
            "region": tile["region"],
            "roi": tile["roi"],
            "tile": tile["tile"],
            "height": h,
            "width": w,
        }
        return out

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, split: str = "train") -> "SEN12MSCRTSDataset":
        ts = cfg.data.get("ts", {})
        phase = "train" if split == "train" else "eval"
        return cls(
            root=cfg.paths.data_root,
            split=split,
            mask_dir=cfg.paths.get("ts_mask_dir", None),
            n_input=int(ts.get("n_input", 3)),
            optical_bands=list(cfg.data.optical_bands) if cfg.data.get("optical_bands") is not None else None,
            target_policy=ts.get(f"{phase}_target_policy", "least_cloudy"),
            input_policy=ts.get(f"{phase}_input_policy", "first"),
            exclude_target_from_inputs=bool(ts.get("exclude_target_from_inputs", True)),
            clear_thresh=float(ts.get("clear_thresh", 1e-3)),
            min_input_cov=float(ts.get(f"{phase}_min_input_cov", 0.0)),
            max_input_cov=float(ts.get(f"{phase}_max_input_cov", 1.0)),
            val_tiles=list(ts.val_tiles) if ts.get("val_tiles") else None,
            augment=bool(cfg.data.get("augmentation", True)) and split == "train",
            regions=list(ts.regions) if ts.get("regions") else None,
        )
