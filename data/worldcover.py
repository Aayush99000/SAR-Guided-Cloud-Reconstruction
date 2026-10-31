"""
ESA WorldCover land-cover labels for SEN12MS-CR-TS, and the dataset used to
train the frozen land-cover judge.

Labels are fetched once per tile by ``scripts/fetch_worldcover.py`` and stored
as ``<wc_dir>/<train|test>/<region>__<roi>__<tile>_wc.npy`` with shape
(P, H, W) uint8, rows in the same order as the cloud-mask index
(``*_meta.npz`` ``patch_ids``).  Values are class indices 0..10 (see
``WC_CLASSES``) or ``IGNORE_INDEX`` for no data.

WorldCover is an annual map (2020 by default, the closest release to the
2018 SEN12MS-CR-TS imagery), so a label describes the location, not one date.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .sen12mscrts_dataset import SEN12MSCRTSDataset

IGNORE_INDEX = 255

# WorldCover map value -> class name, in class-index order.
WC_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}
CLASS_NAMES = list(WC_CLASSES.values())
NUM_CLASSES = len(CLASS_NAMES)

_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _i, _v in enumerate(WC_CLASSES):
    _LUT[_v] = _i


def wc_to_index(wc: np.ndarray) -> np.ndarray:
    """Map raw WorldCover values (10, 20, ..., 100; 0 = no data) to class indices."""
    return _LUT[wc]


def wc_label_path(wc_dir: str | Path, split: str, tile: Dict) -> Path:
    """Label file for a tile dict from ``SEN12MSCRTSDataset._tiles``."""
    src_split = "test" if split == "test" else "train"
    return Path(wc_dir) / src_split / f"{tile['region']}__{tile['roi']}__{tile['tile']}_wc.npy"


class WorldCoverLabels:
    """Lazy, memory-mapped per-tile label lookup aligned with a SEN12MS-CR-TS dataset."""

    def __init__(self, base: SEN12MSCRTSDataset, wc_dir: str | Path) -> None:
        self.base = base
        self.wc_dir = Path(wc_dir)
        self._cache: Dict[int, np.ndarray] = {}

    def tile(self, t_idx: int) -> np.ndarray:
        arr = self._cache.get(t_idx)
        if arr is None:
            tile = self.base._tiles[t_idx]
            path = wc_label_path(self.wc_dir, self.base.split, tile)
            if not path.exists():
                raise FileNotFoundError(f"No WorldCover labels at {path}; run scripts/fetch_worldcover.py")
            arr = np.load(path, mmap_mode="r")
            if arr.shape[0] != len(tile["patch_ids"]):
                raise ValueError(f"{path}: {arr.shape[0]} rows but {len(tile['patch_ids'])} patches in the index")
            self._cache[t_idx] = arr
        return arr

    def __call__(self, t_idx: int, row: int) -> np.ndarray:
        return np.asarray(self.tile(t_idx)[row])

    def for_sample(self, idx: int) -> np.ndarray:
        """Label for item ``idx`` of the base dataset."""
        return self(*self.base._index[idx])


class LandCoverDataset(Dataset):
    """Clear Sentinel-2 dates paired with WorldCover labels (judge training / validation).

    Uses the tiles, paths and cloud masks of a ``SEN12MSCRTSDataset`` so the
    judge sees exactly the bands and normalisation the bridge predicts.  Only
    dates with coverage <= ``max_cov`` are used; pixels cloudy on that date are
    set to ``IGNORE_INDEX``.

    Args:
        base:    SEN12MS-CR-TS dataset providing the split (train / val / test).
        wc_dir:  Root of the fetched WorldCover labels.
        max_cov: Maximum cloud coverage of a usable date.
        policy:  "random" (one random clear date per draw) or "least_cloudy".
        augment: Random flips / 90-degree rotations.
    """

    def __init__(
        self,
        base: SEN12MSCRTSDataset,
        wc_dir: str | Path,
        max_cov: float = 0.01,
        policy: str = "random",
        augment: bool = False,
    ) -> None:
        if policy not in ("random", "least_cloudy"):
            raise ValueError(f"unknown policy {policy!r}")
        self.base = base
        self.labels = WorldCoverLabels(base, wc_dir)
        self.policy = policy
        self.augment = augment
        self._items = []
        for t_idx, tile in enumerate(base._tiles):
            for row in range(len(tile["patch_ids"])):
                cov = np.where(tile["valid"][row], tile["coverage"][row], np.inf)
                clear = np.flatnonzero(cov <= max_cov)
                if len(clear):
                    self._items.append((t_idx, row, clear, int(np.argmin(cov))))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        t_idx, row, clear, least = self._items[idx]
        t = int(clear[torch.randint(len(clear), (1,)).item()]) if self.policy == "random" else least
        tile = self.base._tiles[t_idx]
        h, w = tile["hw"]
        image = self.base._optical(self.base._s2_path(tile, t, int(tile["patch_ids"][row])))
        label = self.labels(t_idx, row).copy()
        cloudy = np.unpackbits(self.base._masks(t_idx)[row, t], count=h * w).reshape(h, w).astype(bool)
        label[cloudy] = IGNORE_INDEX
        if self.augment:
            k = int(torch.randint(4, (1,)))
            image, label = np.rot90(image, k, axes=(-2, -1)), np.rot90(label, k, axes=(-2, -1))
            if bool(torch.randint(2, (1,))):
                image, label = np.flip(image, axis=-1), np.flip(label, axis=-1)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
        }


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None,
                     num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """(num_classes, num_classes) counts, rows = reference, cols = prediction."""
    keep = target != IGNORE_INDEX
    if valid is not None:
        keep &= valid
    idx = target[keep].long() * num_classes + pred[keep].long()
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def scores_from_confusion(conf: np.ndarray) -> Dict[str, float]:
    """Pixel accuracy, mIoU over classes present in the reference, and per-class IoU."""
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    ref, pred = conf.sum(1), conf.sum(0)
    union = ref + pred - tp
    present = ref > 0
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    out = {
        "pixel_acc": float(tp.sum() / conf.sum()) if conf.sum() else float("nan"),
        "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
        "n_pixels": int(conf.sum()),
    }
    out.update({f"iou_{name}": float(iou[i]) if present[i] else float("nan") for i, name in enumerate(CLASS_NAMES)})
    return out
