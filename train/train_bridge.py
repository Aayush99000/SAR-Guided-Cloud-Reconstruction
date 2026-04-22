"""Diffusion Bridge training script (pixel-space OT-ODE).

Trains SAROpticalUNet as the x₀-prediction network inside an OT-ODE bridge.
SAR (Sentinel-1 VV/VH) guides cloud removal from Sentinel-2 optical imagery.

Usage
-----
    # From project root:
    python train/train_bridge.py
    python train/train_bridge.py training.num_epochs=200 training.batch_size=4
    python train/train_bridge.py --resume outputs/checkpoints/bridge/latest.ckpt

Config overrides use dotlist syntax (key=value).  All keys map to
configs/default.yaml.  --resume is a dedicated flag handled separately.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from omegaconf import OmegaConf
    _OMEGACONF = True
except ImportError:
    _OMEGACONF = False

try:
    import wandb as _wandb_mod
    _WANDB = True
except ImportError:
    _WANDB = False

from data import SEN12MSCRDataset, collate_fn
from models import SAROpticalUNet, DiffusionBridge
from models.bridge.noise_schedule import BridgeNoiseSchedule
from models.cloud_aware_loss import CloudAwareLoss, _ssim_map

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average of model weights)
# ---------------------------------------------------------------------------

class EMA:
    """Shadow copy of model parameters updated with exponential decay.

    After each optimiser step, call ``ema.update(model)`` to blend the new
    weights into the shadow copy.  Use the ``apply`` context manager at
    inference time to temporarily swap the live weights for the EMA weights.

    Args:
        model: The ``nn.Module`` whose parameters are tracked.
        decay: EMA coefficient.  0.9999 is typical for image generation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # Always kept as float32 for numerical stability
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend model parameters into shadow:  shadow = decay·shadow + (1−decay)·param."""
        d = self.decay
        for name, param in model.state_dict().items():
            if name not in self.shadow:
                # New key (e.g. from dynamic module) — just copy
                self.shadow[name] = param.detach().clone().float()
            elif param.is_floating_point():
                self.shadow[name].mul_(d).add_(param.float(), alpha=1.0 - d)
            else:
                # Integer buffers (num_batches_tracked, etc.) — copy directly
                self.shadow[name].copy_(param)

    @contextlib.contextmanager
    def apply(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily replace model weights with EMA weights for inference."""
        # Save current live weights
        original: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        # Load EMA weights, cast to match each parameter's original dtype
        ema_state = {
            k: v.to(dtype=original[k].dtype, device=original[k].device)
            for k, v in self.shadow.items()
            if k in original
        }
        model.load_state_dict(ema_state, strict=False)
        try:
            yield model
        finally:
            model.load_state_dict(original)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone().float() for k, v in state.items()}

    def __repr__(self) -> str:
        return f"EMA(decay={self.decay}, tracked_tensors={len(self.shadow)})"


# ---------------------------------------------------------------------------
# Per-batch metrics
# ---------------------------------------------------------------------------

def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB, assuming inputs in [0, 1]."""
    mse = (pred.float() - target.float()).pow(2).mean()
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse.item())


def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.float() - target.float()).abs().mean().item()


def _ssim_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean SSIM ∈ [−1, 1]; higher is better."""
    return _ssim_map(pred.float(), target.float()).mean().item()


class MetricAccumulator:
    """Running sum → mean over batches.  Thread-unsafe; single-process only."""

    def __init__(self) -> None:
        self._sums:   Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def update(self, d: Dict[str, float]) -> None:
        for k, v in d.items():
            fv = float(v)
            if not math.isfinite(fv):   # skip NaN/Inf — don't poison the mean
                continue
            self._sums[k]   = self._sums.get(k, 0.0) + fv
            self._counts[k] = self._counts.get(k, 0)  + 1

    def mean(self) -> Dict[str, float]:
        return {k: self._sums[k] / max(self._counts[k], 1) for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    bridge: DiffusionBridge,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: EMA,
    metrics: Dict[str, float],
    best_val_psnr: float = -float("inf"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":         epoch,
            "bridge":        bridge.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "scaler":        scaler.state_dict(),
            "ema":           ema.state_dict(),
            "metrics":       metrics,
            "best_val_psnr": best_val_psnr,
        },
        path,
    )
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: Path,
    *,
    bridge: DiffusionBridge,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: EMA,
    device: torch.device,
) -> int:
    """Load a checkpoint; returns the epoch to resume from (saved_epoch + 1)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    bridge.load_state_dict(ckpt["bridge"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    ema.load_state_dict(ckpt["ema"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_val_psnr = float(ckpt.get("best_val_psnr", -float("inf")))
    log.info(
        "Resumed from %s  (saved at epoch %d → starting epoch %d, best PSNR so far: %.2f dB)",
        path, ckpt["epoch"], start_epoch, best_val_psnr,
    )
    return start_epoch, best_val_psnr


# ---------------------------------------------------------------------------
# Model / optimiser builders
# ---------------------------------------------------------------------------

def build_model(cfg, device: torch.device) -> Tuple[DiffusionBridge, EMA]:
    """Construct SAROpticalUNet → BridgeNoiseSchedule → CloudAwareLoss → DiffusionBridge."""
    unet = SAROpticalUNet(
        in_channels_optical=cfg.model.in_channels_optical,
        in_channels_sar=cfg.data.sar_bands,
        base_channels=cfg.model.base_channels,
        channel_mult=list(cfg.model.channel_mult),
        num_nafblocks=list(cfg.model.num_nafblocks),
        num_dec_nafblocks=list(cfg.model.num_dec_nafblocks),
        num_vim_blocks=cfg.model.num_vim_blocks,
        time_emb_dim=cfg.model.time_emb_dim,
        dropout=cfg.model.dropout,
        vim_d_state=cfg.model.vim_d_state,
        bottleneck_type=cfg.model.get("bottleneck_type", "vim"),
        fusion_mode=cfg.model.get("fusion_mode", "sfblock"),
    ).to(device)

    schedule = BridgeNoiseSchedule(
        num_steps=cfg.diffusion.diffusion_steps,
        schedule_type=cfg.diffusion.alpha_schedule_type,
    )

    criterion = CloudAwareLoss(
        alpha=cfg.loss.alpha,
        lambda_mse=cfg.loss.lambda_mse,
        lambda_ssim=cfg.loss.lambda_ssim,
    )

    bridge = DiffusionBridge(
        model=unet,
        noise_schedule=schedule,
        device=device,
        loss_fn=criterion,
        t_low=cfg.diffusion.t_low,
        t_high=cfg.diffusion.t_high,
    ).to(device)

    n_params = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    log.info("DiffusionBridge | %.2f M trainable parameters", n_params / 1e6)

    ema = EMA(bridge, decay=cfg.training.ema_decay)
    return bridge, ema


def build_optimizer(
    cfg, bridge: DiffusionBridge, total_steps: int
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
    )

    warmup = cfg.scheduler.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    train_ds = SEN12MSCRDataset.from_config(cfg, split="train")
    val_ds   = SEN12MSCRDataset.from_config(cfg, split="val")

    _nw = cfg.training.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=_nw,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(_nw > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(_nw > 0),
    )
    log.info(
        "Dataloaders | train=%d batches  val=%d batches",
        len(train_loader), len(val_loader),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    bridge:      DiffusionBridge,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    scaler:      torch.cuda.amp.GradScaler,
    ema:         EMA,
    loader:      DataLoader,
    cfg,
    device:      torch.device,
    epoch:       int,
    global_step: int,
    wandb_run,
) -> Tuple[Dict[str, float], int]:
    """One full pass over the training set.

    Supports gradient accumulation (``cfg.training.accumulate_grad_batches``)
    and mixed-precision autocast.

    Returns:
        (mean_metrics_dict, updated_global_step)
    """
    bridge.train()
    acc        = MetricAccumulator()
    accum      = cfg.training.accumulate_grad_batches
    log_every  = cfg.logging.log_every_n_steps
    use_amp    = cfg.training.mixed_precision and device.type == "cuda"
    # bfloat16 has native hardware support on Ampere+ (A100, L40, RTX30/40xx).
    # V100 (Volta, sm_70) emulates bfloat16 in float32 — use float16 there instead.
    _cc = torch.cuda.get_device_capability(device) if use_amp else (0, 0)
    amp_dtype  = torch.bfloat16 if (_cc[0] >= 8) else (torch.float16 if use_amp else torch.float32)

    optimizer.zero_grad()

    for local_step, batch in enumerate(loader):
        # Move all tensors to device; leave metadata (dicts/lists) on CPU
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # --- Forward + cloud-aware loss ---
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss, metrics = bridge.training_step(batch)
            loss_scaled   = loss / accum       # gradient accumulation scaling

        if not torch.isfinite(loss):
            log.warning("[E%03d step %06d] NaN/Inf loss — skipping batch", epoch, global_step)
            optimizer.zero_grad()
            continue

        scaler.scale(loss_scaled).backward()

        # --- Optimiser step every `accum` mini-batches ---
        if (local_step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(bridge.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            ema.update(bridge)
            global_step += 1

        # --- Logging ---
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        acc.update(metrics)

        if global_step > 0 and global_step % log_every == 0:
            log.info(
                "[E%03d step %06d] loss=%.4f  mse=%.4f  ssim=%.4f  lr=%.2e",
                epoch, global_step,
                metrics.get("total",  float("nan")),
                metrics.get("mse",    float("nan")),
                metrics.get("ssim",   float("nan")),
                metrics["lr"],
            )
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{k}": v for k, v in metrics.items()},
                    step=global_step,
                )

    return acc.mean(), global_step


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    bridge:      DiffusionBridge,
    ema:         EMA,
    loader:      DataLoader,
    cfg,
    device:      torch.device,
    epoch:       int,
    global_step: int,
    wandb_run,
) -> Dict[str, float]:
    """Run inference with EMA weights; compute PSNR, SSIM, MAE over the val set.

    Logs a grid of (cloudy | prediction | clean) sample images to W&B on the
    first batch.

    Returns mean metrics dict.
    """
    use_amp   = cfg.training.mixed_precision and device.type == "cuda"
    _cc       = torch.cuda.get_device_capability(device) if use_amp else (0, 0)
    amp_dtype = torch.bfloat16 if (_cc[0] >= 8) else (torch.float16 if use_amp else torch.float32)
    nfe       = cfg.diffusion.sampler_nfe
    acc       = MetricAccumulator()
    images_logged = False

    with ema.apply(bridge):
        bridge.eval()

        for batch in loader:
            x_cloudy   = batch["cloudy"].to(device, non_blocking=True)
            x_clean    = batch["clean"].to(device, non_blocking=True)
            sar        = batch["sar"].to(device, non_blocking=True)
            cloud_mask = batch.get("cloud_mask")
            if cloud_mask is not None:
                cloud_mask = cloud_mask.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                x_pred = bridge.sample(
                    x_cloudy, sar,
                    cloud_mask=cloud_mask,
                    num_steps=nfe,
                )

            # Float32 for all metrics
            x_pred_f  = x_pred.float().clamp(0.0, 1.0)
            x_clean_f = x_clean.float().clamp(0.0, 1.0)

            acc.update({
                "psnr": _psnr(x_pred_f, x_clean_f),
                "ssim": _ssim_score(x_pred_f, x_clean_f),
                "mae":  _mae(x_pred_f, x_clean_f),
            })

            # Log image samples for first batch only
            if wandb_run is not None and not images_logged:
                _log_images_wandb(
                    wandb_run,
                    x_cloudy=x_cloudy.float(),
                    x_pred=x_pred_f,
                    x_clean=x_clean_f,
                    cloud_mask=cloud_mask,
                    epoch=epoch,
                    global_step=global_step,
                    max_images=cfg.logging.get("val_log_images", 4),
                )
                images_logged = True

    bridge.train()   # restore training mode

    means = acc.mean()
    log.info(
        "[Epoch %03d] val  PSNR=%.2f dB  SSIM=%.4f  MAE=%.4f",
        epoch,
        means.get("psnr", 0.0),
        means.get("ssim", 0.0),
        means.get("mae",  0.0),
    )
    if wandb_run is not None:
        wandb_run.log(
            {f"val/{k}": v for k, v in means.items()},
            step=global_step,
        )
    return means


def _log_images_wandb(
    wandb_run,
    *,
    x_cloudy:   torch.Tensor,
    x_pred:     torch.Tensor,
    x_clean:    torch.Tensor,
    cloud_mask: Optional[torch.Tensor],
    epoch:      int,
    global_step: int,
    max_images: int = 4,
) -> None:
    """Log (cloudy | prediction | clean) triplets as a W&B image panel."""
    import numpy as np

    n = min(x_cloudy.shape[0], max_images)
    panels: List[_wandb_mod.Image] = []

    def to_rgb(t: torch.Tensor, i: int) -> "np.ndarray":
        """First 3 channels → uint8 RGB (H, W, 3)."""
        img = t[i, :3].cpu().float().numpy()          # (3, H, W)
        img = (img.clip(0, 1) * 255).astype("uint8")
        return img.transpose(1, 2, 0)                  # (H, W, 3)

    import numpy as _np
    for i in range(n):
        strip_np = _np.concatenate(
            [to_rgb(x_cloudy, i), to_rgb(x_pred, i), to_rgb(x_clean, i)],
            axis=1,                                    # stack horizontally
        )
        caption = f"idx={i} epoch={epoch} | cloudy → pred → clean"
        panels.append(_wandb_mod.Image(strip_np, caption=caption))

    wandb_run.log({"val/samples": panels}, step=global_step)


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train(cfg) -> None:
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # --- Dataloaders (needed for total_steps computation) ---
    train_loader, val_loader = build_dataloaders(cfg)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.training.accumulate_grad_batches)
    total_steps     = steps_per_epoch * cfg.training.num_epochs

    # --- Model, EMA, optimiser ---
    bridge, ema = build_model(cfg, device)
    optimizer, scheduler = build_optimizer(cfg, bridge, total_steps)

    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    # GradScaler is only needed for float16; bfloat16 doesn't overflow so skip it
    scaler  = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.get_device_capability(device)[0] < 8))

    # --- W&B (optional) ---
    wandb_run = None
    if cfg.logging.use_wandb:
        if _WANDB:
            wandb_run = _wandb_mod.init(
                project=cfg.logging.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
            )
            log.info("W&B run: %s", wandb_run.url)
        else:
            log.warning("use_wandb=true but `wandb` is not installed; skipping.")

    # --- Resume from checkpoint ---
    start_epoch   = 1
    global_step   = 0
    best_val_psnr = -float("inf")
    resume_path   = cfg.training.get("resume")
    if resume_path:
        p = Path(resume_path)
        if p.exists():
            start_epoch, best_val_psnr = load_checkpoint(
                p,
                bridge=bridge, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, ema=ema,
                device=device,
            )
            global_step = (start_epoch - 1) * steps_per_epoch
        else:
            log.warning("Resume checkpoint not found: %s — starting from scratch.", p)

    ckpt_dir = Path(cfg.paths.bridge_ckpt_dir)

    log.info(
        "Training | epochs=%d  steps/epoch=%d  total_steps=%d",
        cfg.training.num_epochs, steps_per_epoch, total_steps,
    )

    # =====================================================================
    # Training loop
    # =====================================================================
    for epoch in range(start_epoch, cfg.training.num_epochs + 1):
        t0 = time.perf_counter()

        train_metrics, global_step = train_one_epoch(
            bridge, optimizer, scheduler, scaler, ema,
            train_loader, cfg, device, epoch, global_step, wandb_run,
        )

        elapsed = time.perf_counter() - t0
        log.info(
            "[Epoch %03d/%03d] train_loss=%.4f  t=%.1fs  step=%d",
            epoch, cfg.training.num_epochs,
            train_metrics.get("total", float("nan")),
            elapsed, global_step,
        )

        # --- Validation ---
        if epoch % cfg.training.val_every_n_epochs == 0:
            val_metrics = validate(
                bridge, ema, val_loader, cfg, device, epoch, global_step, wandb_run,
            )

            # Save best checkpoint by PSNR
            val_psnr = val_metrics.get("psnr", -float("inf"))
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                save_checkpoint(
                    ckpt_dir / "best.ckpt",
                    epoch=epoch,
                    bridge=bridge, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, ema=ema,
                    metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                    best_val_psnr=best_val_psnr,
                )
                log.info("New best PSNR: %.2f dB → saved best.ckpt", best_val_psnr)

        # --- Periodic checkpoint ---
        if epoch % cfg.training.save_every_n_epochs == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.ckpt",
                epoch=epoch,
                bridge=bridge, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, ema=ema,
                metrics=train_metrics,
                best_val_psnr=best_val_psnr,
            )

        # --- Always overwrite latest.ckpt (enables seamless resume) ---
        save_checkpoint(
            ckpt_dir / "latest.ckpt",
            epoch=epoch,
            bridge=bridge, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, ema=ema,
            metrics=train_metrics,
            best_val_psnr=best_val_psnr,
        )

    if wandb_run is not None:
        wandb_run.finish()

    log.info("Training complete. Best val PSNR: %.2f dB", best_val_psnr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """Split known flags from dotlist overrides; leave the rest for OmegaConf."""
    parser = argparse.ArgumentParser(
        description="Train SAR-guided diffusion bridge",
        add_help=True,
    )
    parser.add_argument(
        "--resume", default=None, metavar="CKPT",
        help="Path to a latest.ckpt / epoch_NNNN.ckpt to resume from",
    )
    parser.add_argument(
        "--config", default=None, metavar="YAML",
        help="Extra YAML to merge on top of default.yaml (e.g. an ablation config)",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if not _OMEGACONF:
        raise SystemExit("omegaconf is required: pip install omegaconf")

    args, overrides = _parse_args()

    # --- Load config ---
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Merge standalone paths.yaml if it exists as a sibling file
    paths_cfg = config_path.parent / "paths.yaml"
    if paths_cfg.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(paths_cfg))

    # Merge extra config (e.g. ablation override file)
    if args.config:
        extra = Path(args.config)
        if not extra.exists():
            raise FileNotFoundError(f"--config file not found: {extra}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))

    # Apply CLI dotlist overrides  (e.g.  training.batch_size=4)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # --resume takes precedence over config key
    if args.resume:
        OmegaConf.update(cfg, "training.resume", args.resume, merge=True)

    log.info("Effective config:\n%s", OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
