"""Train the frozen land-cover judge on real, clear SEN12MS-CR-TS images.

The judge learns WorldCover classes from clear Sentinel-2 dates of the train
tiles; the bridge's val tiles are held out for judge validation and the test
tiles are never used.  The best checkpoint (val mIoU) is what
``train/evaluate_judge.py`` loads.

    python train/train_judge.py --config configs/sen12mscrts.yaml \
        --wc-dir Dataset/SEN12MSCRTS/worldcover --out outputs/checkpoints/judge/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from data.sen12mscrts_dataset import SEN12MSCRTSDataset
from data.worldcover import (
    CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, LandCoverDataset, confusion_matrix, scores_from_confusion,
)
from models.judge import LandCoverJudge

log = logging.getLogger(__name__)


def _load_cfg(config_args, overrides):
    root = Path(__file__).resolve().parent.parent / "configs"
    cfg = OmegaConf.load(root / "default.yaml")
    if (root / "paths.yaml").exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "paths.yaml"))
    # --config is nargs="+", so it swallows dotlist overrides that follow it
    overrides = [c for c in config_args if "=" in c] + overrides
    for extra in (c for c in config_args if "=" not in c):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _base(cfg, split: str) -> SEN12MSCRTSDataset:
    ts = cfg.data.get("ts", {})
    return SEN12MSCRTSDataset(
        cfg.paths.data_root, split=split, mask_dir=cfg.paths.get("ts_mask_dir", None),
        optical_bands=list(cfg.data.optical_bands),
        val_tiles=list(ts.val_tiles) if ts.get("val_tiles") else None,
        augment=False,
    )


@torch.no_grad()
def _band_stats(ds: LandCoverDataset, n: int = 512):
    idx = torch.randperm(len(ds))[:n].tolist()
    x = torch.stack([ds[i]["image"] for i in idx])               # (n, C, H, W)
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        conf += confusion_matrix(pred, y)
    model.train()
    return scores_from_confusion(conf.cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--wc-dir", required=True)
    ap.add_argument("--out", required=True, help="Path of the best checkpoint")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-cov", type=float, default=0.01, help="Max cloud coverage of a training date")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (smoke test)")
    args, overrides = ap.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    cfg = _load_cfg(args.config, overrides)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LandCoverDataset(_base(cfg, "train"), args.wc_dir, args.max_cov, policy="random", augment=True)
    val_ds = LandCoverDataset(_base(cfg, "val"), args.wc_dir, args.max_cov, policy="least_cloudy")
    if args.max_steps:
        val_ds = Subset(val_ds, range(min(len(val_ds), 4 * args.batch_size)))
    log.info("Judge data: %d train / %d val locations with a date <= %.0f%% cloud",
             len(train_ds), len(val_ds), 100 * args.max_cov)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    model = LandCoverJudge(NUM_CLASSES, in_channels=len(cfg.data.optical_bands), encoder=args.encoder)
    mean, std = _band_stats(train_ds)
    model.set_normalization(mean, std)
    log.info("Band mean %s  std %s", mean.numpy().round(4).tolist(), std.numpy().round(4).tolist())
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.max_steps or args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x), y, ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                log.info("[E%02d step %06d] loss=%.4f lr=%.2e  %.1f steps/s",
                         epoch, step, loss.item(), sched.get_last_lr()[0], step / (time.time() - t0))
            if args.max_steps and step >= args.max_steps:
                break

        scores = validate(model, val_loader, device)
        log.info("[Epoch %02d] val  pixel_acc=%.4f  mIoU=%.4f", epoch, scores["pixel_acc"], scores["miou"])
        if scores["miou"] > best:
            best = scores["miou"]
            torch.save({"model": model.state_dict(), "config": model.config, "epoch": epoch,
                        "val": scores, "classes": CLASS_NAMES, "args": vars(args)}, out)
            with open(out.with_suffix(".json"), "w") as f:
                json.dump({"epoch": epoch, **scores}, f, indent=2)
            log.info("New best (mIoU %.4f) → %s", best, out)
        if args.max_steps and step >= args.max_steps:
            break

    log.info("Done: best val mIoU %.4f in %.1f min", best, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
