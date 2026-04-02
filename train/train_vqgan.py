"""VQ-GAN training script.

Usage:
    python train/train_vqgan.py              # uses configs/default.yaml
    python train/train_vqgan.py training.lr=1e-4 vqgan.latent_dim=256
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from omegaconf import OmegaConf, DictConfig
    import hydra
    _HYDRA = True
except ImportError:
    _HYDRA = False

from data import SEN12MSCRDataset
from data.preprocessing import PreprocessTransform
from models.vqgan.encoder import VQGANEncoder
from models.vqgan.decoder import VQGANDecoder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discriminator (simple PatchGAN for adversarial training)
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 13) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_vqgan(cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training VQ-GAN on {device}")

    # --- Data ---
    transform = PreprocessTransform()
    train_ds = SEN12MSCRDataset.from_config(cfg, split="train")
    train_ds.transform = transform
    val_ds = SEN12MSCRDataset.from_config(cfg, split="val")
    val_ds.transform = transform

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    # --- Models ---
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

    discriminator = PatchDiscriminator(in_channels=cfg.data.num_bands).to(device)

    # --- Optimisers ---
    opt_ae = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
    )
    opt_disc = torch.optim.AdamW(
        discriminator.parameters(),
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.optimizer.betas),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)

    ckpt_dir = Path(cfg.paths.vqgan_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    for epoch in range(1, cfg.training.num_epochs + 1):
        encoder.train()
        decoder.train()
        discriminator.train()

        for step, batch in enumerate(train_loader):
            x = batch["s2_clear"].to(device)          # (B, C, H, W)

            with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
                z_q, _, commit_loss = encoder(x)
                x_rec = decoder(z_q)

                # Reconstruction loss
                rec_loss = nn.functional.mse_loss(x_rec, x)

                # Adversarial (generator side)
                fake_logits = discriminator(x_rec)
                g_loss = nn.functional.binary_cross_entropy_with_logits(
                    fake_logits, torch.ones_like(fake_logits)
                )

                ae_loss = rec_loss + commit_loss + cfg.loss.lambda_adversarial * g_loss

            opt_ae.zero_grad()
            scaler.scale(ae_loss).backward()
            scaler.unscale_(opt_ae)
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                cfg.training.grad_clip,
            )
            scaler.step(opt_ae)

            # Discriminator update
            with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
                real_logits = discriminator(x.detach())
                fake_logits = discriminator(x_rec.detach())
                d_loss = 0.5 * (
                    nn.functional.binary_cross_entropy_with_logits(
                        real_logits, torch.ones_like(real_logits)
                    )
                    + nn.functional.binary_cross_entropy_with_logits(
                        fake_logits, torch.zeros_like(fake_logits)
                    )
                )

            opt_disc.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(opt_disc)
            scaler.update()

            if step % cfg.logging.log_every_n_steps == 0:
                log.info(
                    f"[Epoch {epoch}/{cfg.training.num_epochs}] step={step} "
                    f"rec={rec_loss:.4f} commit={commit_loss:.4f} "
                    f"g={g_loss:.4f} d={d_loss:.4f}"
                )

        # Checkpoint
        if epoch % cfg.training.save_every_n_epochs == 0:
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "opt_ae": opt_ae.state_dict(),
                    "epoch": epoch,
                },
                ckpt_dir / f"vqgan_epoch{epoch:04d}.ckpt",
            )
            log.info(f"Saved checkpoint at epoch {epoch}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _HYDRA:
        @hydra.main(config_path="../configs", config_name="default", version_base=None)
        def main(cfg: DictConfig) -> None:
            train_vqgan(cfg)
        main()
    else:
        raise SystemExit("Install hydra-core: pip install hydra-core omegaconf")
