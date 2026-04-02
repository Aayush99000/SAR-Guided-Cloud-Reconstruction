"""SEN12MS-CR dataset loader.

Expected directory layout
-------------------------
<data_root>/
    <scene_id>/
        s1/         # Sentinel-1 GRD (VV, VH)  — .tif
        s2_cloudy/  # cloudy Sentinel-2 L1C     — .tif
        s2_cloud_free/  # reference clear image — .tif
        cloud_mask/     # binary cloud mask      — .tif  (optional)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError:
    rasterio = None  # type: ignore


class SEN12MSCRDataset(Dataset):
    """PyTorch Dataset for SEN12MS-CR cloud removal.

    Args:
        root:        Path to dataset root directory.
        split_file:  Text file listing scene/patch IDs (one per line).
        patch_size:  Spatial size of each patch (square).
        num_s2_bands: Number of Sentinel-2 bands to load (default 13).
        transform:   Optional callable applied to the returned dict.
        use_cloud_mask: Whether to load/generate cloud masks.
        cloud_threshold: Min cloud-pixel fraction to consider a sample "cloudy".
    """

    S1_SUBDIR = "s1"
    S2_CLOUDY_SUBDIR = "s2_cloudy"
    S2_CLEAR_SUBDIR = "s2_cloud_free"
    MASK_SUBDIR = "cloud_mask"

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        patch_size: int = 256,
        num_s2_bands: int = 13,
        transform: Optional[Callable] = None,
        use_cloud_mask: bool = True,
        cloud_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.patch_size = patch_size
        self.num_s2_bands = num_s2_bands
        self.transform = transform
        self.use_cloud_mask = use_cloud_mask
        self.cloud_threshold = cloud_threshold

        self.samples: List[str] = self._load_split(split_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_split(split_file: str | Path) -> List[str]:
        split_file = Path(split_file)
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            return [line.strip() for line in f if line.strip()]

    def _read_tif(self, path: Path) -> np.ndarray:
        """Read a GeoTIFF and return a (C, H, W) float32 array."""
        if rasterio is None:
            raise ImportError("rasterio is required: pip install rasterio")
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (C, H, W)
        return data

    def _center_crop(self, arr: np.ndarray) -> np.ndarray:
        """Center-crop (C, H, W) to (C, patch_size, patch_size)."""
        _, h, w = arr.shape
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        return arr[:, top : top + self.patch_size, left : left + self.patch_size]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_id = self.samples[idx]
        scene_dir = self.root / scene_id

        # --- Sentinel-1 (VV + VH) ---
        s1_path = scene_dir / self.S1_SUBDIR / f"{scene_id}_s1.tif"
        s1 = self._center_crop(self._read_tif(s1_path))[:2]  # keep VV, VH

        # --- Cloudy Sentinel-2 ---
        s2c_path = scene_dir / self.S2_CLOUDY_SUBDIR / f"{scene_id}_s2_cloudy.tif"
        s2_cloudy = self._center_crop(self._read_tif(s2c_path))[: self.num_s2_bands]

        # --- Cloud-free reference ---
        s2f_path = scene_dir / self.S2_CLEAR_SUBDIR / f"{scene_id}_s2_cloudfree.tif"
        s2_clear = self._center_crop(self._read_tif(s2f_path))[: self.num_s2_bands]

        sample: Dict[str, torch.Tensor] = {
            "s1": torch.from_numpy(s1),
            "s2_cloudy": torch.from_numpy(s2_cloudy),
            "s2_clear": torch.from_numpy(s2_clear),
            "scene_id": scene_id,
        }

        # --- Cloud mask ---
        if self.use_cloud_mask:
            mask_path = scene_dir / self.MASK_SUBDIR / f"{scene_id}_mask.tif"
            if mask_path.exists():
                mask = self._center_crop(self._read_tif(mask_path))[0:1]  # (1, H, W)
                sample["cloud_mask"] = torch.from_numpy(mask)
            else:
                # fall back to SCL-based estimation from the cloudy image
                from .cloud_mask import generate_cloud_mask
                sample["cloud_mask"] = generate_cloud_mask(sample["s2_cloudy"])

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Convenience factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, split: str = "train") -> "SEN12MSCRDataset":
        """Instantiate from an OmegaConf / Hydra config object."""
        split_file = getattr(cfg.paths, f"{split}_list")
        return cls(
            root=cfg.paths.data_root,
            split_file=split_file,
            patch_size=cfg.data.patch_size,
            num_s2_bands=cfg.data.num_bands,
            cloud_threshold=cfg.data.cloud_threshold,
        )
