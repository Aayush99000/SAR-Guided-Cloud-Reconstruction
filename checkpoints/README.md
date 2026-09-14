# Checkpoints

Checkpoint files (`.ckpt`) are excluded from git due to size (~2 GB each).

## v3 + LPIPS (bridge_mt_v3)

| File | Epoch | Val PSNR | Notes |
|------|-------|----------|-------|
| `best.ckpt` | 48 | 24.47 dB | Best validation PSNR — use this for inference/eval |
| `epoch_0100.ckpt` | 100 | — | Final epoch |

**Cluster paths:**
- Home (persistent): `/home/katoch.aa/SAR-Diffusion-Bridge/checkpoints/best.ckpt`
- Scratch (full history): `/scratch/katoch.aa/SAR-Guided-Cloud-Reconstruction/outputs/checkpoints/bridge_mt_v3/`

## Download

```bash
scp katoch.aa@login.explorer.northeastern.edu:/home/katoch.aa/SAR-Diffusion-Bridge/checkpoints/best.ckpt checkpoints/
```
