"""SEN12MS-CR PyTorch Dataset.

Expected directory layout
-------------------------
<data_root>/
    s1/           # Sentinel-1 GRD patches — .tif  (bands: VV, VH)
    s2/           # Sentinel-2 cloud-free reference — .tif  (13 bands)
    s2_cloudy/    # Sentinel-2 cloud-contaminated — .tif   (13 bands)
    splits/
        train.csv
        val.csv
        test.csv

CSV format (one row per patch)
-------------------------------
Required columns:
    s1          relative path from data_root, e.g. "s1/ROIs1158_spring_s1_1.tif"
    s2_clean    relative path, e.g. "s2/ROIs1158_spring_s2_1.tif"
    s2_cloudy   relative path, e.g. "s2_cloudy/ROIs1158_spring_s2_cloudy_1.tif"
Optional columns:
    patch_id         human-readable identifier (falls back to row index)
    cloud_coverage   float in [0, 1]; used for optional filtering
    season           e.g. "spring"
    roi              e.g. "ROIs1158"

Multi-temporal SAR
------------------
When multitemporal=True (default), the dataset builds a lookup from
(roi, patch) → {season: s1_path} across all four seasons.  For each
sample the SAR stack is:

    cat(SAR_spring, SAR_summer, SAR_fall, SAR_winter)  →  (8, H, W)

Missing seasons are zero-padded.  The season order is always fixed
(spring → summer → fall → winter) so the model sees a consistent
channel layout regardless of which season is the target.

Set multitemporal=False or data.multitemporal_sar=false in the config
to fall back to single-date SAR (2 channels) for ablation studies.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    from rasterio.errors import RasterioIOError
except ImportError:
    rasterio = None  # type: ignore
    RasterioIOError = OSError  # type: ignore

from .cloud_mask import generate_cloud_mask, cloud_coverage_fraction

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel-2 band ordering (13-band L1C / L2A stack, 0-indexed)
# ---------------------------------------------------------------------------
# 0:B1  1:B2  2:B3  3:B4  4:B5  5:B6  6:B7  7:B8  8:B8A  9:B9  10:B10  11:B11  12:B12
S2_RGB_NIR_BANDS: Tuple[int, ...] = (1, 2, 3, 7)   # B2, B3, B4, B8

# Fixed season order — determines channel layout in multi-temporal stack
SEASONS: Tuple[str, ...] = ("spring", "summer", "fall", "winter")

# Regex to strip "_{season}_s1" from a SAR filename stem
# e.g. "ROIs1970_fall_s1_114_p107" → "ROIs1970_114_p107"
_SEASON_S1_RE = re.compile(
    r'_(spring|summer|fall|winter)_s1', re.IGNORECASE
)

# Per-band SAR clip ranges (dB, empirical percentiles on SEN12MS-CR)
_SAR_CLIP: Dict[int, Tuple[float, float]] = {
    0: (-25.0, 0.0),    # VV
    1: (-32.5, 0.0),    # VH
}
_S2_CLIP_MIN = 0.0
_S2_CLIP_MAX = 10000.0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _mt_patch_key(s1_path: str) -> str:
    """Return a season-agnostic patch key from an s1 file path.

    Strips the ``_{season}_s1`` segment from the filename stem so that
    all four seasonal acquisitions of the same location share one key.

    Example::

        "s1/ROIs1970_fall_s1_114_p107.tif"  →  "ROIs1970_114_p107"
        "s1/ROIs1158_spring_s1_101_p205.tif" →  "ROIs1158_101_p205"
    """
    stem = Path(s1_path).stem          # drop directory and .tif
    return _SEASON_S1_RE.sub('', stem, count=1)


def _preprocess_sar(arr: np.ndarray) -> np.ndarray:
    """Per-band clip → [0, 1] scale.

    Args:
        arr: (C, H, W) float32 SAR array in dB.

    Returns:
        (C, H, W) float32 in [0, 1].
    """
    out = np.empty_like(arr)
    for band_idx in range(arr.shape[0]):
        lo, hi = _SAR_CLIP.get(band_idx, (-25.0, 0.0))
        clipped = np.clip(arr[band_idx], lo, hi)
        out[band_idx] = (clipped - lo) / (hi - lo)
    return out


def _preprocess_s2(arr: np.ndarray) -> np.ndarray:
    """Clip [0, 10000] → scale to [0, 1].

    Args:
        arr: (C, H, W) float32 S2 array in surface reflectance (×10000).

    Returns:
        (C, H, W) float32 in [0, 1].
    """
    arr = np.clip(arr, _S2_CLIP_MIN, _S2_CLIP_MAX)
    return arr / _S2_CLIP_MAX


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SEN12MSCRDataset(Dataset):
    """PyTorch Dataset for SEN12MS-CR cloud removal.

    Args:
        root:
            Root directory containing the ``s1/``, ``s2/``, and ``s2_cloudy/``
            sub-directories.
        split:
            One of ``"train"``, ``"val"``, or ``"test"``.  The matching
            ``splits/<split>.csv`` is read automatically.
        split_csv:
            Explicit path to a CSV file.  Overrides *split* when provided.
        optical_bands:
            Sequence of 0-based band indices to load from the S2 stacks.
            Defaults to ``(1, 2, 3, 7)`` — B2 (Blue), B3 (Green), B4 (Red),
            B8 (NIR).  Pass ``None`` to load all 13 bands.
        multitemporal:
            If True (default), stack SAR from all four seasons →  (8, H, W).
            Missing seasons are zero-padded.  Set False to use single-date
            SAR (2 channels) for ablation studies.
        transform:
            Optional callable applied to the returned sample dict after all
            preprocessing.  Receives and must return the dict.
        min_cloud_coverage:
            If > 0, patches with ``cloud_coverage < min_cloud_coverage`` are
            excluded (only effective when the CSV contains a ``cloud_coverage``
            column).
        max_cloud_coverage:
            If < 1, patches with ``cloud_coverage > max_cloud_coverage`` are
            excluded.
        dilate_mask:
            Number of pixels to morphologically dilate the cloud mask.
            0 disables dilation.
    """

    _SPLIT_DIR = "splits"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        split_csv: str | Path | None = None,
        optical_bands: Sequence[int] | None = S2_RGB_NIR_BANDS,
        multitemporal: bool = True,
        transform: Optional[Callable[[dict], dict]] = None,
        min_cloud_coverage: float = 0.0,
        max_cloud_coverage: float = 1.0,
        dilate_mask: int = 0,
    ) -> None:
        super().__init__()
        if rasterio is None:
            raise ImportError(
                "rasterio is required: conda install -c conda-forge rasterio"
            )

        self.root          = Path(root)
        self.optical_bands = list(optical_bands) if optical_bands is not None else None
        self.multitemporal = multitemporal
        self.transform     = transform
        self.dilate_mask   = dilate_mask

        csv_path = Path(split_csv) if split_csv else self.root / self._SPLIT_DIR / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        # Filtered records — used for __getitem__
        self.records: List[dict] = self._load_csv(
            csv_path, min_cloud_coverage, max_cloud_coverage
        )

        # Multi-temporal lookup: patch_key → {season: s1_path}
        # Built from ALL rows (no cloud-coverage filter) so every season's
        # SAR is available as conditioning even when that season was filtered out.
        if multitemporal:
            self._mt_lookup: Dict[str, Dict[str, str]] = self._build_mt_lookup(csv_path)
            n_full = sum(len(v) == 4 for v in self._mt_lookup.values())
            log.info(
                "SEN12MSCRDataset | split=%s | %d patches | MT-SAR=ON "
                "| %d/%d locations have all 4 seasons",
                split, len(self.records), n_full, len(self._mt_lookup),
            )
        else:
            self._mt_lookup = {}
            log.info(
                "SEN12MSCRDataset | split=%s | %d patches | MT-SAR=OFF | bands=%s",
                split, len(self.records), self.optical_bands,
            )

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(
        csv_path: Path,
        min_cc: float,
        max_cc: float,
    ) -> List[dict]:
        records = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                for col in ("s1", "s2_clean", "s2_cloudy"):
                    if col not in row:
                        raise ValueError(
                            f"CSV missing required column '{col}' in {csv_path}"
                        )
                cc = float(row["cloud_coverage"]) if "cloud_coverage" in row else None
                if cc is not None:
                    if cc < min_cc or cc > max_cc:
                        continue
                row["_row_idx"] = i
                row["_cloud_coverage_csv"] = cc
                records.append(row)

        if not records:
            log.warning("No records loaded from %s (after coverage filtering).", csv_path)
        return records

    @staticmethod
    def _build_mt_lookup(csv_path: Path) -> Dict[str, Dict[str, str]]:
        """Build {patch_key: {season: s1_path}} from ALL rows in the CSV.

        No cloud-coverage filtering — we want every season's SAR available
        as conditioning regardless of whether that season was cloudy.
        """
        lookup: Dict[str, Dict[str, str]] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                season  = row.get("season", "").lower().strip()
                s1_path = row.get("s1", "").strip()
                if not season or not s1_path or season not in SEASONS:
                    continue
                key = _mt_patch_key(s1_path)
                if key not in lookup:
                    lookup[key] = {}
                lookup[key][season] = s1_path
        return lookup

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        p = Path(rel_path)
        if p.is_absolute():
            return p
        return self.root / p

    def _read_tif(self, path: Path) -> Tuple[np.ndarray, dict]:
        try:
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
                meta = {
                    "crs":       str(src.crs),
                    "transform": src.transform,
                    "width":     src.width,
                    "height":    src.height,
                }
            return data, meta
        except (RasterioIOError, FileNotFoundError) as exc:
            log.error("Failed to read %s: %s", path, exc)
            raise

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | dict]:
        rec = self.records[idx]

        s1_path  = self._resolve(rec["s1"])
        s2c_path = self._resolve(rec["s2_cloudy"])
        s2f_path = self._resolve(rec["s2_clean"])

        # ------------------------------------------------------------------
        # Load current-season SAR (always needed)
        # ------------------------------------------------------------------
        try:
            s1_arr, s1_meta = self._read_tif(s1_path)
        except (RasterioIOError, FileNotFoundError):
            log.warning("Missing S1 file: %s — skipping to next sample.", s1_path)
            return self.__getitem__((idx + 1) % len(self))

        s1_arr = _preprocess_sar(s1_arr)           # (2, H, W) in [0, 1]
        h, w   = s1_arr.shape[1], s1_arr.shape[2]

        # ------------------------------------------------------------------
        # Multi-temporal SAR stacking  (spring, summer, fall, winter order)
        # ------------------------------------------------------------------
        if self.multitemporal:
            key          = _mt_patch_key(rec["s1"])
            season_paths = self._mt_lookup.get(key, {})
            cur_season   = rec.get("season", "").lower().strip()

            bands: List[np.ndarray] = []
            for season in SEASONS:
                if season == cur_season:
                    # Use already-loaded and preprocessed array
                    bands.append(s1_arr)
                elif season in season_paths:
                    path = self._resolve(season_paths[season])
                    try:
                        arr, _ = self._read_tif(path)
                        bands.append(_preprocess_sar(arr))   # (2, H, W)
                    except Exception:
                        bands.append(np.zeros((2, h, w), dtype=np.float32))
                else:
                    # Season not available — zero-pad
                    bands.append(np.zeros((2, h, w), dtype=np.float32))

            sar = torch.from_numpy(np.concatenate(bands, axis=0))  # (8, H, W)
        else:
            sar = torch.from_numpy(s1_arr)                         # (2, H, W)

        # ------------------------------------------------------------------
        # Load & preprocess cloud-contaminated S2
        # ------------------------------------------------------------------
        try:
            s2c_arr, s2c_meta = self._read_tif(s2c_path)
        except (RasterioIOError, FileNotFoundError):
            log.warning("Missing S2-cloudy file: %s — skipping.", s2c_path)
            return self.__getitem__((idx + 1) % len(self))

        s2c_arr = _preprocess_s2(s2c_arr)
        if self.optical_bands is not None:
            s2c_arr = s2c_arr[self.optical_bands]
        cloudy = torch.from_numpy(s2c_arr)

        # ------------------------------------------------------------------
        # Load & preprocess cloud-free S2 reference
        # ------------------------------------------------------------------
        try:
            s2f_arr, s2f_meta = self._read_tif(s2f_path)
        except (RasterioIOError, FileNotFoundError):
            log.warning("Missing S2-clean file: %s — skipping.", s2f_path)
            return self.__getitem__((idx + 1) % len(self))

        s2f_arr = _preprocess_s2(s2f_arr)
        if self.optical_bands is not None:
            s2f_arr = s2f_arr[self.optical_bands]
        clean = torch.from_numpy(s2f_arr)

        # ------------------------------------------------------------------
        # Cloud mask
        # ------------------------------------------------------------------
        cloud_mask = generate_cloud_mask(cloudy, brightness_threshold=0.35)

        if self.dilate_mask > 0:
            from .cloud_mask import dilate_cloud_mask
            cloud_mask = dilate_cloud_mask(cloud_mask, dilation_pixels=self.dilate_mask)

        # ------------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------------
        patch_id = rec.get("patch_id") or str(rec["_row_idx"])
        cc_csv   = rec["_cloud_coverage_csv"]
        cc_live  = cloud_coverage_fraction(cloud_mask)

        metadata: dict = {
            "patch_id":       patch_id,
            "cloud_coverage": cc_csv if cc_csv is not None else cc_live,
            "season":         rec.get("season", ""),
            "roi":            rec.get("roi", ""),
            "s1_path":        str(s1_path),
            "s2_cloudy_path": str(s2c_path),
            "s2_clean_path":  str(s2f_path),
            "crs":            s2f_meta["crs"],
            "height":         s2f_meta["height"],
            "width":          s2f_meta["width"],
        }

        sample: Dict[str, torch.Tensor | dict] = {
            "sar":        sar,          # (8, H, W) MT or (2, H, W) single-date
            "cloudy":     cloudy,       # (C, H, W)
            "clean":      clean,        # (C, H, W)
            "cloud_mask": cloud_mask,   # (1, H, W)
            "metadata":   metadata,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, split: str = "train") -> "SEN12MSCRDataset":
        """Instantiate from an OmegaConf / Hydra config object."""
        return cls(
            root=cfg.paths.data_root,
            split=split,
            optical_bands=(
                list(cfg.data.optical_bands)
                if hasattr(cfg.data, "optical_bands")
                else S2_RGB_NIR_BANDS
            ),
            multitemporal=getattr(cfg.data, "multitemporal_sar", True),
            min_cloud_coverage=getattr(cfg.data, "min_cloud_coverage", 0.0),
            max_cloud_coverage=getattr(cfg.data, "max_cloud_coverage", 1.0),
            dilate_mask=getattr(cfg.data, "dilate_mask", 0),
        )


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    """Custom collate: stacks tensors normally, collects metadata into a list."""
    tensor_keys  = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    metadata_key = "metadata"

    collated = {k: torch.stack([s[k] for s in batch]) for k in tensor_keys}
    collated[metadata_key] = [s[metadata_key] for s in batch]
    return collated
