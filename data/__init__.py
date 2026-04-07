from .sen12mscr_dataset import SEN12MSCRDataset, collate_fn, S2_RGB_NIR_BANDS
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
