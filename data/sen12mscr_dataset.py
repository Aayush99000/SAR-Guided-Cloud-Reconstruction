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
"""

from __future__ import annotations

import csv
import logging
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

# Per-band SAR clip ranges (dB, empirical percentiles on SEN12MS-CR)
_SAR_CLIP: Dict[int, Tuple[float, float]] = {
    0: (-25.0, 0.0),    # VV
    1: (-32.5, 0.0),    # VH
}
_S2_CLIP_MIN = 0.0
_S2_CLIP_MAX = 10000.0


# ---------------------------------------------------------------------------
# Low-level helpers (numpy, pre-tensor)
# ---------------------------------------------------------------------------

def _preprocess_sar(arr: np.ndarray) -> np.ndarray:
    """Per-band clip → positive shift → [0, 1] scale.

    Args:
        arr: (C, H, W) float32 SAR array in dB.

    Returns:
        (C, H, W) float32 in [0, 1].
    """
    out = np.empty_like(arr)
    for band_idx in range(arr.shape[0]):
        lo, hi = _SAR_CLIP.get(band_idx, (-25.0, 0.0))
        clipped = np.clip(arr[band_idx], lo, hi)
        out[band_idx] = (clipped - lo) / (hi - lo)   # → [0, 1]
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

        self.root = Path(root)
        self.optical_bands = list(optical_bands) if optical_bands is not None else None
        self.transform = transform
        self.dilate_mask = dilate_mask

        csv_path = Path(split_csv) if split_csv else self.root / self._SPLIT_DIR / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        self.records: List[dict] = self._load_csv(
            csv_path, min_cloud_coverage, max_cloud_coverage
        )
        log.info(
            "SEN12MSCRDataset | split=%s | %d patches | bands=%s",
            split,
            len(self.records),
            self.optical_bands,
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
                # Validate required columns
                for col in ("s1", "s2_clean", "s2_cloudy"):
                    if col not in row:
                        raise ValueError(
                            f"CSV missing required column '{col}' in {csv_path}"
                        )
                # Optional cloud_coverage filter
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

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a path that may be absolute or relative to self.root."""
        p = Path(rel_path)
        if p.is_absolute():
            return p
        return self.root / p

    def _read_tif(self, path: Path) -> Tuple[np.ndarray, dict]:
        """Read a GeoTIFF → (C, H, W) float32 array + spatial metadata dict."""
        try:
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)   # (C, H, W)
                meta = {
                    "crs": str(src.crs),
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
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

        s1_path      = self._resolve(rec["s1"])
        s2c_path     = self._resolve(rec["s2_cloudy"])
        s2f_path     = self._resolve(rec["s2_clean"])

        # ------------------------------------------------------------------
        # Load & preprocess Sentinel-1
        # ------------------------------------------------------------------
        try:
            s1_arr, s1_meta = self._read_tif(s1_path)
        except (RasterioIOError, FileNotFoundError):
            log.warning("Missing S1 file: %s — skipping to next sample.", s1_path)
            return self.__getitem__((idx + 1) % len(self))

        s1_arr = _preprocess_sar(s1_arr)           # (2, H, W) in [0, 1]
        sar = torch.from_numpy(s1_arr)             # float32

        # ------------------------------------------------------------------
        # Load & preprocess cloud-contaminated S2
        # ------------------------------------------------------------------
        try:
            s2c_arr, s2c_meta = self._read_tif(s2c_path)
        except (RasterioIOError, FileNotFoundError):
            log.warning("Missing S2-cloudy file: %s — skipping.", s2c_path)
            return self.__getitem__((idx + 1) % len(self))

        s2c_arr = _preprocess_s2(s2c_arr)          # (13, H, W) in [0, 1]
        if self.optical_bands is not None:
            s2c_arr = s2c_arr[self.optical_bands]  # (C_sel, H, W)
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
        # Cloud mask (generated on-the-fly from cloudy optical bands)
        # ------------------------------------------------------------------
        # generate_cloud_mask expects (C, H, W) in [0, 1]; use blue (B2 = idx 0
        # in the selected band stack) as the brightness proxy.
        cloud_mask = generate_cloud_mask(
            cloudy,
            brightness_threshold=0.35,
        )                                          # (1, H, W) float32

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
            "sar":        sar,          # (2, H, W)    float32  [0, 1]
            "cloudy":     cloudy,       # (C, H, W)    float32  [0, 1]
            "clean":      clean,        # (C, H, W)    float32  [0, 1]
            "cloud_mask": cloud_mask,   # (1, H, W)    float32  {0, 1}
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
            min_cloud_coverage=getattr(cfg.data, "min_cloud_coverage", 0.0),
            max_cloud_coverage=getattr(cfg.data, "max_cloud_coverage", 1.0),
            dilate_mask=getattr(cfg.data, "dilate_mask", 0),
        )


# ---------------------------------------------------------------------------
# Collate helper — strips metadata dicts so DataLoader default_collate works
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    """Custom collate: stacks tensors normally, collects metadata into a list."""
    tensor_keys  = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    metadata_key = "metadata"

    collated = {k: torch.stack([s[k] for s in batch]) for k in tensor_keys}
    collated[metadata_key] = [s[metadata_key] for s in batch]
    return collated
