"""Diffusion Bridge training script.

Requires a pretrained VQ-GAN encoder/decoder (see train_vqgan.py).
Trains the U-Net velocity field + SAR fusion blocks end-to-end.

Usage:
    python train/train_bridge.py
    python train/train_bridge.py training.num_epochs=50 diffusion.diffusion_steps=500
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from omegaconf import DictConfig, OmegaConf
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
from models.cloud_aware_loss import CloudAwareLoss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAR encoder (lightweight — maps SAR VV/VH → latent feature map)
# ---------------------------------------------------------------------------

class SAREncoder(nn.Module):
    """Minimal CNN to project Sentinel-1 (2-band) into a latent feature map."""

    def __init__(self, sar_bands: int = 2, out_channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(sar_bands, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, 3, padding=1, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------

def load_vqgan(cfg, device: torch.device):
    encoder = VQGANEncoder(
        in_channels=cfg.data.num_bands,
        channels=list(cfg.vqgan.encoder_channels),
        latent_dim=cfg.vqgan.latent_dim,
        num_res_blocks=cfg.vqgan.num_res_blocks,
        num_embeddings=cfg.vqgan.num_embeddings,
        commitment_cost=cfg.vqgan.commitment_cost,
    ).to(device)

    decoder = VQGANDecoder(
        out_channels=cfg.data.num_bands,
        channels=list(cfg.vqgan.decoder_channels),
        latent_dim=cfg.vqgan.latent_dim,
        num_res_blocks=cfg.vqgan.num_res_blocks,
    ).to(device)

    if cfg.paths.pretrained_vqgan:
        ckpt = torch.load(cfg.paths.pretrained_vqgan, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        log.info(f"Loaded VQ-GAN from {cfg.paths.pretrained_vqgan}")

    # Freeze VQ-GAN during bridge training
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    return encoder.eval(), decoder.eval()


def train_bridge(cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training Diffusion Bridge on {device}")

    # --- Data ---
    transform = PreprocessTransform()
    train_ds = SEN12MSCRDataset.from_config(cfg, split="train")
    train_ds.transform = transform
    val_ds = SEN12MSCRDataset.from_config(cfg, split="val")
    val_ds.transform = transform

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, num_workers=4)

    # --- VQ-GAN (frozen) ---
    vqgan_enc, vqgan_dec = load_vqgan(cfg, device)

    # --- SAR encoder ---
    sar_enc = SAREncoder(sar_bands=cfg.data.sar_bands, out_channels=cfg.vqgan.latent_dim).to(device)

    # --- Velocity U-Net ---
    unet = UNet(
        in_channels=cfg.vqgan.latent_dim,
        cond_channels=cfg.vqgan.latent_dim,
        base_channels=cfg.unet.base_channels,
        channel_multipliers=list(cfg.unet.channel_multipliers),
        num_res_blocks=cfg.unet.num_res_blocks,
        dropout=cfg.unet.dropout,
    ).to(device)

    # --- Diffusion bridge ---
    bridge = DiffusionBridge(
        velocity_net=unet,
        schedule_name=cfg.diffusion.alpha_schedule_type,
        diffusion_steps=cfg.diffusion.diffusion_steps,
        ot_reg=cfg.diffusion.ot_reg,
    ).to(device)

    # --- Loss ---
    criterion = CloudAwareLoss(
        cloud_loss_alpha=cfg.loss.cloud_loss_alpha,
        lambda_mse=cfg.loss.lambda_mse,
        lambda_ssim=cfg.loss.lambda_ssim,
        lambda_perceptual=cfg.loss.lambda_perceptual,
    )

    # --- Optimiser ---
    trainable = list(bridge.parameters()) + list(sar_enc.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.num_epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)

    ckpt_dir = Path(cfg.paths.bridge_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    # --- Training loop ---
    for epoch in range(1, cfg.training.num_epochs + 1):
        bridge.train()
        sar_enc.train()

        for step, batch in enumerate(train_loader):
            s1 = batch["s1"].to(device)
            s2_cloudy = batch["s2_cloudy"].to(device)
            s2_clear = batch["s2_clear"].to(device)
            mask = batch.get("cloud_mask", None)
            if mask is not None:
                mask = mask.to(device)

            with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
                # Encode to latent space (no grad through VQ-GAN)
                with torch.no_grad():
                    z0, _, _ = vqgan_enc(s2_cloudy)   # source latent
                    z1, _, _ = vqgan_enc(s2_clear)    # target latent

                sar_feat = sar_enc(s1)

                # Bridge flow-matching loss
                bridge_loss, bridge_metrics = bridge.training_loss(z0, z1, cond=sar_feat)

                # Pixel-space consistency loss via VQ-GAN decoder
                sampler = ODESampler(
                    velocity_fn=bridge.predict_velocity,
                    method="euler",
                    num_steps=cfg.diffusion.sampler_nfe,
                )
                with torch.no_grad():
                    z_rec = sampler.sample(z0, cond=sar_feat)
                    x_rec = vqgan_dec(z_rec)

                pixel_loss, pixel_metrics = criterion(x_rec, s2_clear, cloud_mask=mask)
                loss = bridge_loss + 0.1 * pixel_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if step % cfg.logging.log_every_n_steps == 0:
                log.info(
                    f"[Epoch {epoch}/{cfg.training.num_epochs}] step={step} "
                    f"bridge={bridge_loss:.4f} pixel={pixel_loss:.4f}"
                )

        scheduler.step()

        # --- Validation ---
        if epoch % cfg.training.val_every_n_epochs == 0:
            val_loss = _validate(bridge, sar_enc, vqgan_enc, vqgan_dec, criterion, val_loader, cfg, device)
            log.info(f"[Epoch {epoch}] val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"bridge": bridge.state_dict(), "sar_enc": sar_enc.state_dict(), "epoch": epoch},
                    ckpt_dir / "best.ckpt",
                )
                log.info("Saved best checkpoint.")

        if epoch % cfg.training.save_every_n_epochs == 0:
            torch.save(
                {"bridge": bridge.state_dict(), "sar_enc": sar_enc.state_dict(), "epoch": epoch},
                ckpt_dir / f"bridge_epoch{epoch:04d}.ckpt",
            )


@torch.no_grad()
def _validate(bridge, sar_enc, vqgan_enc, vqgan_dec, criterion, loader, cfg, device) -> float:
    bridge.eval()
    sar_enc.eval()
    total = 0.0

    sampler = ODESampler(
        velocity_fn=bridge.predict_velocity,
        method="euler",
        num_steps=cfg.diffusion.sampler_nfe,
    )

    for batch in loader:
        s1 = batch["s1"].to(device)
        s2_cloudy = batch["s2_cloudy"].to(device)
        s2_clear = batch["s2_clear"].to(device)
        mask = batch.get("cloud_mask", None)
        if mask is not None:
            mask = mask.to(device)

        z0, _, _ = vqgan_enc(s2_cloudy)
        sar_feat = sar_enc(s1)
        z_rec = sampler.sample(z0, cond=sar_feat)
        x_rec = vqgan_dec(z_rec)

        loss, _ = criterion(x_rec, s2_clear, cloud_mask=mask)
        total += loss.item()

    return total / max(len(loader), 1)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _HYDRA:
        @hydra.main(config_path="../configs", config_name="default", version_base=None)
        def main(cfg: DictConfig) -> None:
            train_bridge(cfg)
        main()
    else:
        raise SystemExit("Install hydra-core: pip install hydra-core omegaconf")
