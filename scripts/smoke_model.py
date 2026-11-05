"""Smoke test for a model variant on real SEN12MS-CR-TS batches.

1. The v1 checkpoint still loads and samples with the default config.
2. A fresh bridge built from the given overlays trains for a few steps (timing
   and peak memory at the training batch size) and samples, and its
   prediction responds to the cloud mask and, for temporal models, to the
   input dates.

    PYTHONPATH=. python scripts/smoke_model.py --v1-ckpt outputs/checkpoints/bridge_ts_v1/best.ckpt \
        --config configs/sen12mscrts.yaml configs/sen12mscrts_temporal.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.evaluate import load_model                        # noqa: E402
from train.train_bridge import build_dataloaders, build_model  # noqa: E402
from train.train_judge import _load_cfg                       # noqa: E402


def _to(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-ckpt", default="")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    device = torch.device("cuda")
    loader_args = ["training.num_workers=4", f"training.batch_size={args.batch_size}"]

    cfg = _load_cfg(args.config, loader_args)
    train_loader, val_loader = build_dataloaders(cfg)
    val = _to(next(iter(val_loader)), device)
    x_c, sar, cm = val["cloudy"], val["sar"], val["cloud_mask"]
    print(f"[batch] cloudy {tuple(x_c.shape)} sar {tuple(sar.shape)} stack {tuple(val['cloudy_stack'].shape)} "
          f"dt {val['input_dt'][0].tolist()} target_doy {val['target_doy'][0]:.3f}")

    # 1. Regression: the existing v1 checkpoint loads with the default config
    if args.v1_ckpt:
        cfg_v1 = _load_cfg(["configs/sen12mscrts.yaml"], loader_args)
        v1 = load_model(Path(args.v1_ckpt), cfg_v1, device).eval()
        with torch.no_grad():
            pred = v1.sample(x_c, sar, cloud_mask=cm, num_steps=3, composite=False,
                             temporal=v1.temporal_inputs(val))
        print(f"[v1] sample {tuple(pred.shape)} OK")
        del v1

    # 2. Variant: a few training steps, then sampling
    bridge, _ = build_model(cfg, device)
    m = bridge.model
    n_par = sum(p.numel() for p in m.parameters())
    print(f"[model] mask_input={m.mask_input} temporal_mode={m.temporal_mode} "
          f"date_conditioning={m.date_conditioning} "
          f"opt_input in_ch={m.opt_input.in_channels} params={n_par / 1e6:.2f}M")
    opt = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    bridge.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for step, batch in zip(range(args.steps), train_loader):
        if step == 2:                       # skip warm-up steps in the timing
            torch.cuda.synchronize()
            t0 = time.time()
        loss, _ = bridge.training_step(_to(batch, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"[train] step {step:2d} loss {loss.item():.4f}")
    torch.cuda.synchronize()
    print(f"[train] {(time.time() - t0) / (args.steps - 2):.3f} s/step incl. loading at batch {args.batch_size}, "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if m.temporal_mode == "attention":
        g = sum(f.proj.weight.abs().sum().item() for f in m.temporal_fusion)
        print(f"[train] |temporal fusion proj| after training {g:.3e}")
        assert g > 0, "temporal fusion stayed at its zero initialisation"
    if m.date_conditioning:
        g = m.date_emb.mlp[-1].weight.abs().sum().item()
        print(f"[train] |date embedding output layer| after training {g:.3e}")
        assert g > 0, "date embedding stayed at its zero initialisation"

    bridge.eval()
    temporal = bridge.temporal_inputs(val)
    with torch.no_grad():
        p = bridge.sample(x_c, sar, cloud_mask=cm, num_steps=3, composite=False, temporal=temporal)
        assert p.shape == x_c.shape and torch.isfinite(p).all()
        if m.mask_input:
            p0 = bridge.sample(x_c, sar, cloud_mask=torch.zeros_like(cm), num_steps=3, composite=False,
                               temporal=temporal)
            d = (p - p0).abs().mean().item()
            print(f"[sample] mean |pred(mask) - pred(zero mask)| = {d:.3e}")
            assert d > 0, "prediction ignores the cloud mask"
        if m.temporal_mode != "none":
            shifted = {**temporal, "input_dt": temporal["input_dt"] - 0.5}
            p1 = bridge.sample(x_c, sar, cloud_mask=cm, num_steps=3, composite=False, temporal=shifted)
            d = (p - p1).abs().mean().item()
            print(f"[sample] mean |pred(dates) - pred(input dates - 6 months)| = {d:.3e}")
            assert d > 0, "prediction ignores the input dates"
        if m.date_conditioning:
            season = {**temporal, "target_doy": (temporal["target_doy"] + 0.5) % 1.0}
            p2 = bridge.sample(x_c, sar, cloud_mask=cm, num_steps=3, composite=False, temporal=season)
            d = (p - p2).abs().mean().item()
            print(f"[sample] mean |pred(target date) - pred(target date + 6 months)| = {d:.3e}")
            assert d > 0, "prediction ignores the target date"
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
