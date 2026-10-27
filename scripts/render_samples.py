"""
Render comparison strips for selected test samples without a full evaluation.

Usage:
    python scripts/render_samples.py --config configs/sen12mscrts.yaml \\
        --ckpt outputs/checkpoints/bridge_ts_v1/best.ckpt --nfe 3 \\
        --count 20 --out-dir outputs/eval/ts_v1/ours/strips_nfe3 [dotlist overrides]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import build_dataset, collate_fn          # noqa: E402
from train.evaluate import load_model, save_comparison_strip  # noqa: E402
from utils.metrics import compute_metrics            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, nargs="+", help="YAML(s) merged in order on top of default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--nfe", type=int, default=3)
    ap.add_argument("--count", type=int, default=20, help="Evenly spaced samples across the split")
    ap.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit dataset indices")
    ap.add_argument("--out-dir", required=True)
    args, overrides = ap.parse_known_args()

    root = Path(__file__).resolve().parent.parent / "configs"
    cfg = OmegaConf.load(root / "default.yaml")
    if (root / "paths.yaml").exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "paths.yaml"))
    for extra in args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bridge = load_model(Path(args.ckpt), cfg, device)
    ds = build_dataset(cfg, split=args.split)
    idx = args.indices or [round(i * (len(ds) - 1) / max(args.count - 1, 1)) for i in range(args.count)]
    loader = DataLoader(Subset(ds, idx), batch_size=4, num_workers=4, collate_fn=collate_fn)
    composite = cfg.data.get("composite_output", True)
    out_dir = Path(args.out_dir)

    k = 0
    with torch.no_grad():
        for batch in loader:
            x_c, x_0, sar, cm = (batch[n].to(device) for n in ("cloudy", "clean", "sar", "cloud_mask"))
            pred = bridge.sample(x_c, sar, cloud_mask=cm, num_steps=args.nfe, composite=composite).float().clamp(0, 1)
            for i in range(pred.shape[0]):
                m = compute_metrics(pred[i:i + 1], x_0[i:i + 1].float())
                meta = batch["metadata"][i]
                name = f"sample_{idx[k]:05d}.png"
                save_comparison_strip(
                    sar=sar[i], cloudy=x_c[i], pred=pred[i], clean=x_0[i],
                    save_path=out_dir / name, cloud_mask=cm[i:i + 1],
                    psnr_val=m["psnr"], ssim_val=m["ssim"], sam_val=m["sam"],
                    coverage=meta.get("cloud_coverage"),
                )
                dates = f"inputs {meta.get('input_dates')} -> target {meta.get('target_date')}" if "target_date" in meta else ""
                print(f"{name}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.3f}  {meta.get('patch_id', '')}  {dates}", flush=True)
                k += 1


if __name__ == "__main__":
    main()
