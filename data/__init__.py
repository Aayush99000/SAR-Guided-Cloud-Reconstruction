from .sen12mscr_dataset import SEN12MSCRDataset, collate_fn, S2_RGB_NIR_BANDS
from .sen12mscrts_dataset import SEN12MSCRTSDataset


def build_dataset(cfg, split: str = "train"):
    """Instantiate the dataset named by ``cfg.data.dataset`` (default sen12mscr)."""
    name = cfg.data.get("dataset", "sen12mscr")
    if name == "sen12mscr":
        return SEN12MSCRDataset.from_config(cfg, split=split)
    if name == "sen12mscrts":
        return SEN12MSCRTSDataset.from_config(cfg, split=split)
    raise ValueError(f"Unknown data.dataset {name!r} (expected 'sen12mscr' or 'sen12mscrts')")

from .preprocessing import (
    # Primary API
    normalize_optical,
    normalize_sar,
    to_diffusion_range,
    from_diffusion_range,
    extract_rgb,
    patchify,
    unpatchify,
    PreprocessTransform,
    # Backwards-compatible aliases
    clip_and_normalize,
    normalize_s2,
    normalize_s1,
    extract_patches,
    reconstruct_from_patches,
)
from .cloud_mask import (
    generate_cloud_mask,
    cloud_thickness_weight,
    cloud_coverage_fraction,
    dilate_cloud_mask,
    apply_cloud_mask,
)

__all__ = [
    # Dataset
    "SEN12MSCRDataset",
    "SEN12MSCRTSDataset",
    "build_dataset",
    "collate_fn",
    "S2_RGB_NIR_BANDS",
    # Preprocessing — primary
    "normalize_optical",
    "normalize_sar",
    "to_diffusion_range",
    "from_diffusion_range",
    "extract_rgb",
    "patchify",
    "unpatchify",
    "PreprocessTransform",
    # Preprocessing — aliases
    "clip_and_normalize",
    "normalize_s2",
    "normalize_s1",
    "extract_patches",
    "reconstruct_from_patches",
    # Cloud mask
    "generate_cloud_mask",
    "cloud_thickness_weight",
    "cloud_coverage_fraction",
    "dilate_cloud_mask",
    "apply_cloud_mask",
]
