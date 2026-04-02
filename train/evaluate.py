"""Evaluation script — computes PSNR, SSIM, MAE, SAM, LPIPS on the test split.

Usage:
    python train/evaluate.py paths.best_ckpt=outputs/checkpoints/bridge/best.ckpt
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from omegaconf import DictConfig
    import hydra
    _HYDRA = True
except ImportError:
    _HYDRA = False

from data import SEN12MSCRDataset
from data.preprocessing import PreprocessTransform
from models.vqgan.encoder import VQGANEncoder
from models.vqgan.decoder import VQGANDecoder
from models.backbone.unet import UNet
from models.bridge.diffusion_bridge import DiffusionBridge
from models.bridge.sampler import ODESampler
from train.train_bridge import SAREncoder
from utils.metrics import compute_metrics, MetricAggregator
from utils.visualization import save_comparison_grid

log = logging.getLogger(__name__)


def evaluate(cfg) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Evaluating on {device}")

    # --- Data ---
    test_ds = SEN12MSCRDataset.from_config(cfg, split="test")
    test_ds.transform = PreprocessTransform()
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, num_workers=4)

    # --- Load models ---
    vqgan_enc = VQGANEncoder(
        in_channels=cfg.data.num_bands,
        channels=list(cfg.vqgan.encoder_channels),
        latent_dim=cfg.vqgan.latent_dim,
    ).to(device).eval()

    vqgan_dec = VQGANDecoder(
        out_channels=cfg.data.num_bands,
        channels=list(cfg.vqgan.decoder_channels),
        latent_dim=cfg.vqgan.latent_dim,
    ).to(device).eval()

    unet = UNet(
        in_channels=cfg.vqgan.latent_dim,
        cond_channels=cfg.vqgan.latent_dim,
        base_channels=cfg.unet.base_channels,
    ).to(device).eval()

    bridge = DiffusionBridge(
        velocity_net=unet,
        schedule_name=cfg.diffusion.alpha_schedule_type,
    ).to(device).eval()

    sar_enc = SAREncoder(
        sar_bands=cfg.data.sar_bands,
        out_channels=cfg.vqgan.latent_dim,
    ).to(device).eval()

    # Load bridge checkpoint
    ckpt = torch.load(cfg.paths.best_ckpt, map_location=device)
    bridge.load_state_dict(ckpt["bridge"])
    sar_enc.load_state_dict(ckpt["sar_enc"])
    log.info(f"Loaded checkpoint from {cfg.paths.best_ckpt} (epoch {ckpt.get('epoch', '?')})")

    if cfg.paths.pretrained_vqgan:
        vq_ckpt = torch.load(cfg.paths.pretrained_vqgan, map_location=device)
        vqgan_enc.load_state_dict(vq_ckpt["encoder"])
        vqgan_dec.load_state_dict(vq_ckpt["decoder"])

    sampler = ODESampler(
        velocity_fn=bridge.predict_velocity,
        method="euler",
        num_steps=cfg.diffusion.sampler_nfe,
    )

    agg = MetricAggregator()
    viz_dir = Path(cfg.paths.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            s1 = batch["s1"].to(device)
            s2_cloudy = batch["s2_cloudy"].to(device)
            s2_clear = batch["s2_clear"].to(device)
            mask = batch.get("cloud_mask")
            if mask is not None:
                mask = mask.to(device)

            z0, _, _ = vqgan_enc(s2_cloudy)
            sar_feat = sar_enc(s1)
            z_rec = sampler.sample(z0, cond=sar_feat)
            x_rec = vqgan_dec(z_rec)

            metrics = compute_metrics(x_rec, s2_clear, cloud_mask=mask)
            agg.update(metrics)

            # Save a visual sample every 50 batches
            if batch_idx % 50 == 0:
                save_comparison_grid(
                    cloudy=s2_cloudy[0],
                    reconstructed=x_rec[0],
                    clear=s2_clear[0],
                    save_path=viz_dir / f"sample_{batch_idx:04d}.png",
                )

    final_metrics = agg.compute()
    log.info("Test metrics:")
    for k, v in final_metrics.items():
        log.info(f"  {k}: {v:.4f}")

    metrics_path = Path(cfg.paths.metrics_dir)
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(metrics_path / "test_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    log.info(f"Saved metrics to {metrics_path / 'test_metrics.json'}")

    return final_metrics


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _HYDRA:
        @hydra.main(config_path="../configs", config_name="default", version_base=None)
        def main(cfg: DictConfig) -> None:
            evaluate(cfg)
        main()
    else:
        raise SystemExit("Install hydra-core: pip install hydra-core omegaconf")
