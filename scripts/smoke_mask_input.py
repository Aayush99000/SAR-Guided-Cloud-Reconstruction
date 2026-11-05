"""Smoke test for model.mask_input on real SEN12MS-CR-TS batches.

1. The v1 checkpoint still loads and samples with the default config (mask_input=false).
2. A fresh mask_input=true bridge trains for a few steps and samples, and its
   prediction changes when the cloud mask changes.

    PYTHONPATH=. python scripts/smoke_mask_input.py --v1-ckpt outputs/checkpoints/bridge_ts_v1/best.ckpt
"""

import argparse
from pathlib import Path

import torch

from train.evaluate import load_model
from train.train_bridge import build_dataloaders, build_model
from train.train_judge import _load_cfg as _load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-ckpt", required=True)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    device = torch.device("cuda")

    cfg_v1 = _load_config(["configs/sen12mscrts.yaml"], ["training.num_workers=4", "training.batch_size=2"])
    cfg_m = _load_config(["configs/sen12mscrts.yaml", "configs/sen12mscrts_mask.yaml"],
                         ["training.num_workers=4", "training.batch_size=2"])
    assert not cfg_v1.model.mask_input and cfg_m.model.mask_input
    train_loader, val_loader = build_dataloaders(cfg_m)
    val = next(iter(val_loader))
    x_c, sar, cm = (val[k].to(device) for k in ("cloudy", "sar", "cloud_mask"))
    print(f"[batch] cloudy {tuple(x_c.shape)} sar {tuple(sar.shape)} mask {tuple(cm.shape)} "
          f"mask mean {cm.mean():.3f}")

    # 1. Regression: the existing v1 checkpoint loads with the default config
    v1 = load_model(Path(args.v1_ckpt), cfg_v1, device).eval()
    assert v1.model.opt_input.in_channels == 2 * 4 + 6
    with torch.no_grad():
        pred = v1.sample(x_c, sar, cloud_mask=cm, num_steps=3, composite=False)
    print(f"[v1] opt_input in_ch {v1.model.opt_input.in_channels}  sample {tuple(pred.shape)} OK")
    del v1

    # 2. mask_input bridge: a few training steps, then sampling
    bridge, _ = build_model(cfg_m, device)
    assert bridge.model.opt_input.in_channels == 2 * 4 + 6 + 1
    opt = torch.optim.AdamW(bridge.parameters(), lr=1e-4)
    bridge.train()
    for step, batch in zip(range(args.steps), train_loader):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        loss, _ = bridge.training_step(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = bridge.model.opt_input.weight.grad[:, -1].abs().sum().item()
        opt.step()
        if step % 5 == 0:
            print(f"[mask] step {step:2d} loss {loss.item():.4f}  |grad| on mask channel {g:.3e}")
    assert g > 0, "no gradient reaches the mask channel"

    bridge.eval()
    with torch.no_grad():
        p1 = bridge.sample(x_c, sar, cloud_mask=cm, num_steps=3, composite=False)
        p0 = bridge.sample(x_c, sar, cloud_mask=torch.zeros_like(cm), num_steps=3, composite=False)
    d = (p1 - p0).abs().mean().item()
    print(f"[mask] sample {tuple(p1.shape)}  mean |pred(mask) - pred(zero mask)| = {d:.3e}")
    assert d > 0, "prediction ignores the cloud mask"
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
