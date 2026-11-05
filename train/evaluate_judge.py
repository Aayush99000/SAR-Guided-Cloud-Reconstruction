"""Frozen-judge evaluation: does the reconstruction recover the real land cover?

Runs the bridge on the SEN12MS-CR-TS test split, then the frozen land-cover
judge (``train/train_judge.py``) on three images per sample:

    pred    the bridge reconstruction
    cloudy  the cloud-aware input composite (no reconstruction; lower bound)
    clean   the real cloud-free target (upper bound when scored against WorldCover)

and compares the land-cover maps:

    pred_vs_judgeGT    judge(pred)   vs judge(clean)   label-free agreement
    cloudy_vs_judgeGT  judge(cloudy) vs judge(clean)
    pred_vs_wc         judge(pred)   vs ESA WorldCover
    cloudy_vs_wc       judge(cloudy) vs ESA WorldCover
    clean_vs_wc        judge(clean)  vs ESA WorldCover  (judge accuracy on real data)

each over all pixels and over cloud pixels (never clear in any input), for the
whole test set, per target season, and for targets in the same / a different
season than the inputs.  Scores come from summed confusion matrices.

    python train/evaluate_judge.py --config configs/sen12mscrts.yaml \
        --ckpt outputs/checkpoints/bridge_ts_v1/best.ckpt \
        --judge outputs/checkpoints/judge/best.ckpt \
        --wc-dir Dataset/SEN12MSCRTS/worldcover --nfe 3 --out-dir outputs/eval/judge_ts_v1
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import build_dataset, collate_fn
from data.sen12mscrts_dataset import _SEASON_BY_MONTH, SEN12MSCRTSDataset
from data.worldcover import CLASS_NAMES, NUM_CLASSES, WorldCoverLabels, confusion_matrix, scores_from_confusion
from models.judge import LandCoverJudge
from train.evaluate import load_model
from train.train_judge import _load_cfg

log = logging.getLogger(__name__)

# (name, estimate, reference)
_COMPARISONS = [
    ("pred_vs_judgeGT", "pred", "judge_clean"),
    ("cloudy_vs_judgeGT", "cloudy", "judge_clean"),
    ("pred_vs_wc", "pred", "wc"),
    ("cloudy_vs_wc", "cloudy", "wc"),
    ("clean_vs_wc", "clean", "wc"),
]
_PIXELS = ("all", "cloud")


def _season(d: str) -> str:
    return _SEASON_BY_MONTH[date.fromisoformat(d).month]


def _groups(meta: dict) -> list:
    same = meta["season"] in {_season(d) for d in meta["input_dates"]}
    return ["all", meta["season"], "same_season" if same else "diff_season"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True, help="Bridge checkpoint")
    ap.add_argument("--judge", required=True, help="Judge checkpoint from train_judge.py")
    ap.add_argument("--wc-dir", required=True)
    ap.add_argument("--nfe", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N test samples (smoke test)")
    args, overrides = ap.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    cfg = _load_cfg(args.config, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.training.mixed_precision) and device.type == "cuda"

    test_ds = build_dataset(cfg, split="test")
    if not isinstance(test_ds, SEN12MSCRTSDataset):
        raise SystemExit("The judge evaluation needs data.dataset=sen12mscrts")
    labels = WorldCoverLabels(test_ds, args.wc_dir)
    ds = Subset(test_ds, range(min(args.limit, len(test_ds)))) if args.limit else test_ds
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.training.num_workers,
                        pin_memory=True, collate_fn=collate_fn)

    bridge = load_model(Path(args.ckpt), cfg, device)
    judge = LandCoverJudge.from_checkpoint(args.judge, map_location=device).to(device)
    log.info("Judge: %s (epoch %s)", args.judge, torch.load(args.judge, map_location="cpu",
                                                             weights_only=False).get("epoch"))

    conf = defaultdict(lambda: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device))
    counts = defaultdict(int)
    records = []
    k = 0
    with torch.no_grad():
        for batch in loader:
            x_cloudy = batch["cloudy"].to(device, non_blocking=True)
            x_clean = batch["clean"].to(device, non_blocking=True).clamp(0, 1)
            sar = batch["sar"].to(device, non_blocking=True)
            cloud_mask = batch["cloud_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                x_pred = bridge.sample(x_cloudy, sar, cloud_mask=cloud_mask, num_steps=args.nfe,
                                       composite=cfg.data.get("composite_output", True),
                                       temporal=bridge.temporal_inputs(batch))
            x_pred = x_pred.float().clamp(0, 1)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                seg = {name: judge(img).argmax(1)
                       for name, img in (("pred", x_pred), ("cloudy", x_cloudy), ("clean", x_clean))}
            seg["judge_clean"] = seg["clean"]
            bsz = x_pred.shape[0]
            seg["wc"] = torch.from_numpy(np.stack([labels.for_sample(k + i) for i in range(bsz)])).to(device)
            cloud = cloud_mask[:, 0] > 0.5

            for i, meta in enumerate(batch["metadata"]):
                groups = _groups(meta)
                for g in groups:
                    counts[g] += 1
                rec = {"patch_id": meta["patch_id"], "season": meta["season"], "group": groups[2],
                       "target_date": meta["target_date"], "input_dates": " ".join(meta["input_dates"]),
                       "composite_coverage": meta["composite_coverage"]}
                for name, est, ref in _COMPARISONS:
                    for pixels in _PIXELS:
                        c = confusion_matrix(seg[est][i], seg[ref][i], cloud[i] if pixels == "cloud" else None)
                        for g in groups:
                            conf[(name, pixels, g)] += c
                        n = int(c.sum())
                        rec[f"{name}_{pixels}_acc"] = float(c.diag().sum()) / n if n else float("nan")
                records.append(rec)
            k += bsz
            if k % 400 < bsz:
                log.info("%d / %d samples", k, len(ds))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (name, pixels, g), c in sorted(conf.items()):
        rows.append({"comparison": name, "pixels": pixels, "group": g, "n_samples": counts[g],
                     **scores_from_confusion(c.cpu().numpy())})
    with open(out / f"judge_summary_nfe{args.nfe}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / f"judge_per_sample_nfe{args.nfe}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    np.savez(out / f"judge_confusion_nfe{args.nfe}.npz", classes=np.array(CLASS_NAMES),
             **{"|".join(key): c.cpu().numpy() for key, c in conf.items()})

    by = {(r["comparison"], r["pixels"], r["group"]): r for r in rows}
    log.info("Frozen-judge summary (NFE=%d, %d samples) — mIoU / pixel accuracy", args.nfe, len(records))
    log.info("%-18s %-6s %16s %16s %16s", "comparison", "pixels", "all", "same_season", "diff_season")
    for name, _, _ in _COMPARISONS:
        for pixels in _PIXELS:
            cells = []
            for g in ("all", "same_season", "diff_season"):
                r = by.get((name, pixels, g))
                cells.append(f"{r['miou']:.3f} / {r['pixel_acc']:.3f}" if r else "-")
            log.info("%-18s %-6s %16s %16s %16s", name, pixels, *cells)
    log.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
