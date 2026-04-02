# SAR-Guided Optical Cloud Reconstruction via Latent Diffusion Bridge

A pipeline for reconstructing cloud-obscured Sentinel-2 imagery using
Sentinel-1 SAR guidance and an OT-driven latent diffusion bridge.

## Architecture

```
Sentinel-1 (VV/VH) ──► SAREncoder ──────────────────────────────────────────────┐
                                                                                   │ cross-attn
Sentinel-2 cloudy ──► VQ-GAN Encoder (frozen) ──► z_0                            │
                                                    │                             │
                                    Bridge forward  ▼                             │
                                    q(z_t|z_0,z_1)  z_t ──► U-Net ◄─── t-emb ◄──┘
                                                                │
                                    ODE Sampler (1-5 NFE) ◄────┘
                                          │
                                          ▼
                                    VQ-GAN Decoder ──► Reconstructed clear image
```

**Key components:**
- **VQ-GAN** — encodes/decodes 13-band Sentinel-2 patches into a compact latent space
- **OT-Diffusion Bridge** — flow-matching between cloudy and clear latents with sine-based α schedule
- **U-Net backbone** — NAFBlocks + Bidirectional Mamba bottleneck + SAR Fusion Blocks (cross-attention)
- **Cloud-aware loss** — adaptive pixel weighting to prioritise cloud-region reconstruction

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download SEN12MS-CR dataset
bash scripts/download_data.sh --dest /data/SEN12MS-CR

# 3. Run full training pipeline
bash scripts/run_training.sh all

# Or run individual stages:
bash scripts/run_training.sh 1   # VQ-GAN only
bash scripts/run_training.sh 2   # Bridge only (requires VQ-GAN checkpoint)
bash scripts/run_training.sh 3   # Evaluation
```

## Configuration

All hyperparameters live in `configs/default.yaml` and dataset paths in `configs/paths.yaml`.
Override any value on the command line via Hydra syntax:

```bash
python train/train_bridge.py \
  training.batch_size=4 \
  diffusion.alpha_schedule_type=sine \
  diffusion.sampler_nfe=5
```

## Project Layout

```
configs/              Hydra config files
data/                 Dataset & preprocessing
models/
  vqgan/              Encoder + VQ bottleneck + Decoder
  bridge/             OT-ODE diffusion bridge, noise schedule, ODE sampler
  backbone/           NAFBlock, Vision Mamba SSM, SAR Fusion Block, U-Net
  cloud_aware_loss.py Adaptive multi-component loss
train/                Training & evaluation scripts
utils/                Metrics (PSNR, SSIM, MAE, SAM, LPIPS) + visualisation
scripts/              Shell scripts for data download and training
```

## Metrics

| Metric | Description |
|--------|-------------|
| PSNR   | Peak Signal-to-Noise Ratio (dB) |
| SSIM   | Structural Similarity Index |
| MAE    | Mean Absolute Error |
| SAM    | Spectral Angle Mapper (degrees) |
| LPIPS  | Learned Perceptual Image Patch Similarity |

Cloud-region variants (prefixed `cloud_`) evaluate only over masked pixels.

## References

- Ebel et al., *SEN12MS-CR: A Multi-Seasonal Dataset for Cloud Removal*, 2022
- Chen et al., *Simple Baselines for Image Restoration (NAFNet)*, ECCV 2022
- Zhu et al., *Vision Mamba: Efficient Visual Representation Learning*, arXiv 2401.13587
- Liu et al., *Flow Matching for Generative Modeling*, ICLR 2023
