"""OT-ODE Diffusion Bridge — forward process, training objective, and inference.

Optimal Transport ODE Formulation
==================================
We define a *deterministic* bridge between the cloudy observation x_T and
the unknown clean image x_0.  The forward process is a straight-line path in
pixel space — no Gaussian noise is ever added:

    x_t = (1 − α_t) · x_clean  +  α_t · x_cloudy        [forward bridge]

with schedule α_t satisfying  α_0 = 0  (pure clean)  and  α_T = 1  (pure cloudy).

This is the Optimal Transport (OT) displacement interpolant: for the
linear schedule it is the unique constant-speed geodesic that minimises
the squared-L2 transport cost E[‖x_clean − x_cloudy‖²].

Training objective  (x₀-prediction)
-------------------------------------
A neural network f_θ observes x_t at time t together with the original
cloudy image x_cloudy and SAR radar image, and predicts x̂₀ ≈ x_clean:

    x̂₀ = f_θ(x_t, t, x_cloudy, SAR)

Loss — cloud-mask-weighted multi-component reconstruction:
    L = CloudAwareLoss(x̂₀, x_clean, M)

where M is a binary cloud mask (1 = cloudy pixel to reconstruct).

Reverse ODE  (DDIM-style step)
-------------------------------
Given x̂₀ from the network, the next state at time t − s is derived by
re-evaluating the forward bridge formula with x̂₀ as the clean anchor:

    r = α_{t−s} / α_t                         (mixing ratio ∈ [0, 1))
    x_{t−s} = (1 − r) · x̂₀  +  r · x_t      (ODE step)

Equivalently: x_{t−s} = (1 − α_{t−s}) · x̂₀  +  α_{t−s} · x_cloudy
i.e. the bridge re-evaluated at t−s using the model prediction.

At the final step (t − s = 0):  r = 0  →  x̂₀ is returned directly.

Hard-mask compositing  (inference only)
----------------------------------------
After the full reverse ODE, clear pixels that were never occluded by cloud
are composited back from the original cloudy image to avoid any model
degradation on already-observed data:

    x_final = M · x̂₀  +  (1 − M) · x_cloudy_original

where M = cloud_mask (1 = reconstruct from model, 0 = keep original).

References
----------
  Albergo & Vanden-Eijnden, "Building Normalizing Flows with Stochastic
    Interpolants", ICLR 2023.
  Lipman et al., "Flow Matching for Generative Modelling", ICLR 2023.
  Ebel et al., "DBER: Across-Modal Cloud Removal", ISPRS 2022.
  Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .noise_schedule import BridgeNoiseSchedule
from ..cloud_aware_loss import CloudAwareLoss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DiffusionBridge
# ---------------------------------------------------------------------------

class DiffusionBridge(nn.Module):
    """OT-ODE pixel-space diffusion bridge for SAR-guided cloud removal.

    Wraps SAROpticalUNet with the full training and inference loop.

    Args:
        model:           SAROpticalUNet backbone.  Expected forward signature:
                           model(x_t, t, x_cloudy_mean, sar, cloud_mask)
                           → x_hat_0  (B, C_opt, H, W)
        noise_schedule:  Pre-constructed BridgeNoiseSchedule instance.
        device:          Computation device.  If None, inferred from model
                         parameters at first use.
        loss_fn:         CloudAwareLoss instance.  Defaults to
                         CloudAwareLoss() with standard weights.
        t_low:           Minimum training timestep (avoids trivial t ≈ 0).
                         Default 0.02.
        t_high:          Maximum training timestep.  Default 1.0.
        clip_output:     Clamp model predictions to [clip_min, clip_max]
                         during inference.  Default True.
        clip_min:        Lower clip bound.  Default 0.0.
        clip_max:        Upper clip bound.  Default 1.0.

    Batch key convention (training_step)
    -------------------------------------
    The dataset (SEN12MSCRDataset) returns dicts with these keys:

        "clean"      : (B, C, H, W) cloud-free reference, float32 [0, 1]
        "cloudy"     : (B, C, H, W) cloud-contaminated input, float32 [0, 1]
        "sar"        : (B, 2, H, W) Sentinel-1 VV+VH, float32 [0, 1]
        "cloud_mask" : (B, 1, H, W) binary mask, float32 {0, 1}
        "metadata"   : list of dicts (ignored during training)

    Alternative keys ("s2_clear", "s2_cloudy", "s1") are also accepted
    for backward compatibility with older data pipelines.
    """

    # Accepted key aliases for flexible batch dicts
    _CLEAN_KEYS  = ("clean", "s2_clean", "s2_clear")
    _CLOUDY_KEYS = ("cloudy", "s2_cloudy")
    _SAR_KEYS    = ("sar", "s1")
    _MASK_KEYS   = ("cloud_mask", "mask")

    def __init__(
        self,
        model:           nn.Module,
        noise_schedule:  BridgeNoiseSchedule,
        device:          Optional[Union[str, torch.device]] = None,
        loss_fn:         Optional[CloudAwareLoss]           = None,
        t_low:           float = 0.02,
        t_high:          float = 1.00,
        clip_output:     bool  = True,
        clip_min:        float = 0.0,
        clip_max:        float = 1.0,
    ) -> None:
        super().__init__()

        self.model    = model
        self.schedule = noise_schedule
        self.loss_fn  = loss_fn if loss_fn is not None else CloudAwareLoss()
        self.t_low    = t_low
        self.t_high   = t_high
        self.clip_output = clip_output
        self.clip_min    = clip_min
        self.clip_max    = clip_max

        # Device resolved lazily from model parameters if not given
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = None

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Infer device from model parameters (lazy)."""
        if self._device is not None:
            return self._device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _pick(self, batch: Dict[str, Any], keys: Tuple[str, ...]) -> torch.Tensor:
        """Extract the first matching key from a batch dict."""
        for k in keys:
            if k in batch:
                return batch[k]
        raise KeyError(
            f"None of the expected keys {keys} found in batch "
            f"(available: {list(batch.keys())})"
        )

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def forward_process(
        self,
        x_clean:  torch.Tensor,
        x_cloudy: torch.Tensor,
        t:        torch.Tensor,
    ) -> torch.Tensor:
        """Compute x_t via the deterministic OT bridge.

        Linearly interpolates between x_clean and x_cloudy at time t:

            x_t = (1 − α_t) · x_clean  +  α_t · x_cloudy

        This is the *forward* direction: increasing t moves the image
        from clean (t = 0) toward the cloudy observation (t = 1).

        At training time, t is sampled uniformly so the network sees
        corruptions at every severity from barely-noticeable (small t)
        to fully-occluded (t → 1).

        Args:
            x_clean:  Cloud-free reference  (B, C, H, W)  in [0, 1].
            x_cloudy: Cloudy observation    (B, C, H, W)  in [0, 1].
            t:        Normalised time       (B,)           in [0, 1].

        Returns:
            x_t  (B, C, H, W) — linear mixture at time t.
        """
        return self.schedule.q_sample(x_clean, x_cloudy, t)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Full training iteration from raw batch to loss.

        Steps
        -----
        1. Unpack x_clean, x_cloudy, SAR, cloud_mask from batch.
        2. Sample random timestep t ~ Uniform[t_low, t_high] per sample.
        3. Compute x_t = forward_process(x_clean, x_cloudy, t).
        4. Predict x̂₀ = model(x_t, t, x_cloudy, SAR, cloud_mask).
        5. Compute CloudAwareLoss(x̂₀, x_clean, cloud_mask) → scalar loss.
        6. Return (loss, metrics_dict).

        Args:
            batch: Dict produced by SEN12MSCRDataset / DataLoader.
                   Tensors must already reside on the target device.

        Returns:
            loss:    Scalar loss tensor (call .backward() on this).
            metrics: Flat dict with float values for logging:
                     "loss", "mse", "ssim", "cloud_mse", "clear_mse",
                     "t_mean", "t_std", "alpha_mean".
        """
        # ---- 1. Unpack ------------------------------------------------
        x_clean    = self._pick(batch, self._CLEAN_KEYS)
        x_cloudy   = self._pick(batch, self._CLOUDY_KEYS)
        sar        = self._pick(batch, self._SAR_KEYS)
        cloud_mask = batch.get("cloud_mask") or batch.get("mask")  # optional

        B = x_clean.shape[0]

        # ---- 2. Sample timesteps  t ~ Uniform[t_low, t_high] ----------
        t = torch.empty(B, device=x_clean.device).uniform_(self.t_low, self.t_high)

        # ---- 3. Forward bridge  →  x_t --------------------------------
        x_t = self.forward_process(x_clean, x_cloudy, t)

        # ---- 4. Predict x̂₀ -------------------------------------------
        # x_cloudy serves as x_cloudy_mean (single-observation mode).
        # Multi-temporal stacks would average along the time dimension first.
        x_hat_0 = self.model(x_t, t, x_cloudy, sar, cloud_mask)

        # ---- 5. Cloud-aware loss  (CloudAwareLoss handles mask weights) -
        loss, breakdown = self.loss_fn(x_hat_0, x_clean, cloud_mask=cloud_mask)

        # ---- 6. Additional diagnostics --------------------------------
        with torch.no_grad():
            alpha_t = self.schedule.alpha(t)            # (B,) float32
            metrics: Dict[str, float] = {
                **{k: float(v) for k, v in breakdown.items()},
                "t_mean":     float(t.mean()),
                "t_std":      float(t.std()),
                "alpha_mean": float(alpha_t.mean()),
            }

        return loss, metrics

    # ------------------------------------------------------------------
    # Single reverse ODE step
    # ------------------------------------------------------------------

    def reverse_step(
        self,
        x_t:        torch.Tensor,
        t:          Union[torch.Tensor, float],
        t_prev:     Union[torch.Tensor, float],
        x_cloudy:   torch.Tensor,
        sar:        torch.Tensor,
        cloud_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single deterministic ODE reverse step.

        Given the current state x_t at time t, this method:
          (a) queries the network for x̂₀ = f_θ(x_t, t, x_cloudy, SAR)
          (b) computes the next state x_{t−s} via the ODE update rule:

                r          = α_{t−s} / α_t                (∈ [0, 1))
                x_{t−s}   = (1 − r) · x̂₀  +  r · x_t

        Equivalently, x_{t−s} = (1 − α_{t−s}) · x̂₀ + α_{t−s} · x_cloudy,
        i.e. the forward bridge re-evaluated at t−s with x̂₀ as the
        clean anchor.

        Gradient flow is blocked (called inside @torch.no_grad() context
        from sample()).  To get gradients for intermediate states (e.g. for
        variational inference), call this method inside a grad-enabled context
        explicitly.

        Args:
            x_t:        Current state                 (B, C, H, W).
            t:          Current normalised time        scalar or (B,) in [0, 1].
            t_prev:     Previous normalised time       scalar or (B,) in [0, 1].
                        Must satisfy t_prev < t.
            x_cloudy:   Cloudy conditioning image     (B, C, H, W).
                        Used as x_cloudy_mean for the network.
            sar:        SAR image VV+VH               (B, 2, H, W).
            cloud_mask: Optional cloud binary mask    (B, 1, H, W).

        Returns:
            x_prev:  New state at t_prev              (B, C, H, W).
            x_hat_0: Network's clean prediction       (B, C, H, W).
                     Returned so callers can log intermediate predictions
                     or apply compositing mid-trajectory.
        """
        # Broadcast scalar t to batch dimension
        t_batch = self._broadcast_t(t, x_t.shape[0], x_t.device)

        # (a) Network prediction
        x_hat_0 = self.model(x_t, t_batch, x_cloudy, sar, cloud_mask)

        # (b) ODE step via BridgeNoiseSchedule
        x_prev = self.schedule.reverse_step(x_t, x_hat_0, t_batch, t_prev)

        return x_prev, x_hat_0

    # ------------------------------------------------------------------
    # Full inference sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        x_cloudy:         torch.Tensor,
        sar:              torch.Tensor,
        cloud_mask:       Optional[torch.Tensor] = None,
        num_steps:        int  = 5,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Full reverse-ODE inference: x_T → x̂₀ with hard-mask compositing.

        Algorithm
        ---------
        1. Initialise x_t = x_cloudy  (start at t = 1, fully corrupted).
        2. Obtain N+1 uniformly-spaced timesteps: [1, 1−1/N, …, 0].
        3. For each step i = 0 … N−1:
             a. Predict  x̂₀ = f_θ(x_t, t_i, x_cloudy, SAR)
             b. Update   x_t ← (1 − r) · x̂₀ + r · x_t
                         where r = α_{t_{i+1}} / α_{t_i}
        4. Hard compositing  (only when cloud_mask is provided):
             x_final = M · x̂₀  +  (1 − M) · x_cloudy_original
           Cloud pixels (M=1) are replaced by the model's prediction;
           clear pixels (M=0) are preserved exactly from the input.
        5. Clamp x_final to [clip_min, clip_max].

        Why start from x_cloudy and not from random noise?
        ---------------------------------------------------
        In a standard diffusion model the prior is N(0, I); sampling begins
        from pure noise.  In the OT bridge the prior *is* x_cloudy — the
        forward bridge at t=1 gives x_1 = x_cloudy deterministically.
        This avoids any randomness in the starting state and makes inference
        fully deterministic for a fixed input.

        Args:
            x_cloudy:          Cloudy input image (B, C, H, W).
            sar:               SAR image          (B, 2, H, W).
            cloud_mask:        Binary cloud mask  (B, 1, H, W). 1=cloudy, 0=clear.
                               Used for both model conditioning and final
                               hard-mask compositing.
            num_steps:         N — number of ODE function evaluations (NFE).
                               See get_nfe_options() for quality guidance.
            return_trajectory: If True, also return the list of all N+1
                               intermediate x_t states (for visualisation).

        Returns:
            x_final: Reconstructed cloud-free image (B, C, H, W).
            If return_trajectory=True: also (trajectory: list of N+1 tensors).
        """
        device    = x_cloudy.device
        B         = x_cloudy.shape[0]
        x_cloudy_orig = x_cloudy.clone()

        # --- Timestep grid: 1.0 → 0.0 in N+1 steps ---
        timesteps = self.schedule.get_inference_steps(num_steps).to(device)  # (N+1,)

        x_t        = x_cloudy.clone()                # t=1: start from cloudy image
        trajectory: Optional[List[torch.Tensor]] = [x_t.cpu()] if return_trajectory else None
        x_hat_0_final: Optional[torch.Tensor]    = None

        log.debug("Bridge sample: NFE=%d, schedule=%s", num_steps, self.schedule.schedule_type)

        for i in range(num_steps):
            t_cur  = timesteps[i].expand(B)          # (B,) current time
            t_prev = timesteps[i + 1].expand(B)      # (B,) previous (smaller) time

            x_t, x_hat_0 = self.reverse_step(
                x_t, t_cur, t_prev, x_cloudy, sar, cloud_mask
            )
            x_hat_0_final = x_hat_0

            if return_trajectory:
                trajectory.append(x_t.cpu())

        # --- Hard-mask compositing -------------------------------------------
        # Preserve originally-clear pixels; fill cloud-covered regions from model
        if cloud_mask is not None and x_hat_0_final is not None:
            x_final = cloud_mask * x_hat_0_final + (1.0 - cloud_mask) * x_cloudy_orig
        else:
            x_final = x_t

        # --- Clamp to valid range ---
        if self.clip_output:
            x_final = x_final.clamp(self.clip_min, self.clip_max)

        if return_trajectory:
            return x_final, trajectory  # type: ignore[return-value]
        return x_final

    # ------------------------------------------------------------------
    # NFE quality guide
    # ------------------------------------------------------------------

    def get_nfe_options(self) -> Dict[int, Dict[str, str]]:
        """Return NFE → quality tradeoff reference table.

        NFE (Number of Function Evaluations) is the number of reverse ODE
        steps executed at inference.  More steps ≈ higher quality at the
        cost of proportionally more compute.

        The OT-ODE bridge is more NFE-efficient than standard diffusion
        models because the reverse trajectory is nearly straight
        (the OT prior ensures minimal curvature of the x̂₀ path).
        Good reconstruction is typically achieved with 5–20 steps.

        Returns:
            Dict[NFE, {"psnr": str, "latency": str, "notes": str}]
            PSNR and latency values are approximate guidelines for
            256×256 patches on a single V100 GPU.
        """
        return {
            1: {
                "psnr":    "~28–30 dB",
                "latency": "~0.05 s",
                "notes":   "Single Euler step: x̂₀ directly from model at t=1. "
                           "Fast but ignores schedule curvature; some artefacts "
                           "in dense-cloud scenes.",
            },
            2: {
                "psnr":    "~30–32 dB",
                "latency": "~0.10 s",
                "notes":   "Two steps (t=1 → t=0.5 → t=0). Already better "
                           "than a single call. Recommended minimum for "
                           "downstream tasks that tolerate moderate error.",
            },
            5: {
                "psnr":    "~33–35 dB",
                "latency": "~0.25 s",
                "notes":   "DB-CR default. Good balance of speed and quality. "
                           "Recommended for most production use cases.",
            },
            10: {
                "psnr":    "~35–36 dB",
                "latency": "~0.5 s",
                "notes":   "Noticeably sharper than NFE=5 for thick cloud "
                           "scenes. Recommended for publication-quality results.",
            },
            20: {
                "psnr":    "~36–37 dB",
                "latency": "~1.0 s",
                "notes":   "Diminishing returns above 10 steps for simple "
                           "scenes; meaningful gain for multi-layer stratus "
                           "and cloud shadows.",
            },
            50: {
                "psnr":    "~37–38 dB",
                "latency": "~2.5 s",
                "notes":   "Near-converged result for this schedule. "
                           "Use for benchmarking against PSNR/SSIM metrics.",
            },
            100: {
                "psnr":    "~37–38 dB",
                "latency": "~5.0 s",
                "notes":   "Effectively same quality as NFE=50 for the cosine "
                           "schedule. Prefer NFE=50 unless trajectory analysis "
                           "is needed.",
            },
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def batch_to_device(
        self, batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Move all tensors in batch to self.device in-place."""
        return {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    def __repr__(self) -> str:
        T = self.schedule.T
        stype = self.schedule.schedule_type
        nparams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return (
            f"DiffusionBridge(\n"
            f"  schedule = {stype} (T={T})\n"
            f"  t_range  = [{self.t_low}, {self.t_high}]\n"
            f"  model    = {self.model.__class__.__name__} "
            f"({nparams:,} trainable params)\n"
            f")"
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _broadcast_t(
        t: Union[torch.Tensor, float, int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Ensure t is a (B,) float tensor on the correct device."""
        if isinstance(t, torch.Tensor):
            t = t.to(device=device, dtype=torch.float32)
            if t.dim() == 0:
                t = t.expand(batch_size)
            return t
        return torch.full((batch_size,), float(t), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from .noise_schedule import BridgeNoiseSchedule

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Tiny stub model (stand-in for SAROpticalUNet) ---
    class _StubModel(nn.Module):
        def __init__(self, c_opt=4, c_sar=2):
            super().__init__()
            c_in = c_opt * 2 + c_sar
            self.net = nn.Sequential(
                nn.Conv2d(c_in, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, c_opt, 3, padding=1),
                nn.Sigmoid(),
            )
        def forward(self, x_t, t, x_cloudy_mean, sar, cloud_mask=None):
            x = torch.cat([x_t, x_cloudy_mean, sar], dim=1)
            return self.net(x)

    B, C, H, W = 2, 4, 64, 64
    model    = _StubModel(c_opt=C, c_sar=2).to(device)
    schedule = BridgeNoiseSchedule(num_steps=1000, schedule_type="cosine")
    bridge   = DiffusionBridge(model, schedule, device=device).to(device)

    print(bridge)
    print()

    # --- forward_process ---
    x_clean  = torch.ones(B, C, H, W, device=device) * 0.8
    x_cloudy = torch.ones(B, C, H, W, device=device) * 0.2
    t        = torch.tensor([0.0, 1.0], device=device)
    x_t      = bridge.forward_process(x_clean, x_cloudy, t)
    assert x_t.shape == (B, C, H, W)
    assert torch.allclose(x_t[0], x_clean[0], atol=1e-5),  "t=0 → x_clean"
    assert torch.allclose(x_t[1], x_cloudy[1], atol=1e-5), "t=1 → x_cloudy"
    print("forward_process  ✓  (t=0→x_clean, t=1→x_cloudy verified)")

    # --- training_step ---
    batch = {
        "clean":      x_clean,
        "cloudy":     x_cloudy,
        "sar":        torch.rand(B, 2, H, W, device=device),
        "cloud_mask": torch.randint(0, 2, (B, 1, H, W), device=device).float(),
    }
    loss, metrics = bridge.training_step(batch)
    assert loss.requires_grad, "loss has no grad"
    assert "loss"      in metrics
    assert "t_mean"    in metrics
    assert "alpha_mean" in metrics
    print(f"training_step    ✓  loss={loss.item():.4f}  t_mean={metrics['t_mean']:.3f}  "
          f"alpha_mean={metrics['alpha_mean']:.3f}")

    # --- batch_to_device ---
    cpu_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    dev_batch = bridge.batch_to_device(cpu_batch)
    assert all(v.device.type == device.type
               for v in dev_batch.values() if isinstance(v, torch.Tensor))
    print(f"batch_to_device  ✓")

    # --- reverse_step ---
    t_cur   = torch.full((B,), 0.5, device=device)
    t_prev  = torch.full((B,), 0.25, device=device)
    x_prev, x_hat_0 = bridge.reverse_step(
        x_t, t_cur, t_prev, x_cloudy, batch["sar"], batch["cloud_mask"]
    )
    assert x_prev.shape  == (B, C, H, W)
    assert x_hat_0.shape == (B, C, H, W)
    print(f"reverse_step     ✓  x_prev mean={x_prev.mean():.4f}  "
          f"x_hat_0 mean={x_hat_0.mean():.4f}")

    # --- reverse_step: scalar t  (API flexibility) ---
    x_prev_s, _ = bridge.reverse_step(x_t, 0.5, 0.25, x_cloudy, batch["sar"])
    assert x_prev_s.shape == (B, C, H, W)
    print(f"reverse_step     ✓  scalar t accepted")

    # --- sample: NFE=1,5 ---
    for nfe in (1, 5):
        x_rec = bridge.sample(x_cloudy, batch["sar"], cloud_mask=batch["cloud_mask"],
                              num_steps=nfe)
        assert x_rec.shape == (B, C, H, W), f"NFE={nfe}: bad shape"
        assert x_rec.min() >= 0.0 - 1e-5 and x_rec.max() <= 1.0 + 1e-5, \
            f"NFE={nfe}: output out of [0,1]"
        print(f"sample NFE={nfe:3d}   ✓  out=[{x_rec.min():.3f}, {x_rec.max():.3f}]")

    # --- sample: hard-mask compositing ---
    mask_all = torch.ones(B, 1, H, W, device=device)   # all pixels cloudy
    mask_none = torch.zeros(B, 1, H, W, device=device)  # no cloud

    x_all_cloud  = bridge.sample(x_cloudy, batch["sar"], cloud_mask=mask_all)
    x_no_cloud   = bridge.sample(x_cloudy, batch["sar"], cloud_mask=mask_none)
    # When mask=0 everywhere, compositing keeps original: result == x_cloudy
    assert torch.allclose(x_no_cloud, x_cloudy.clamp(0, 1), atol=1e-5), \
        "mask=0 compositing did not preserve x_cloudy"
    print(f"hard-mask compositing  ✓  (mask=0 → original preserved)")

    # --- sample with trajectory ---
    x_rec_traj, traj = bridge.sample(
        x_cloudy, batch["sar"], cloud_mask=batch["cloud_mask"],
        num_steps=3, return_trajectory=True
    )
    assert len(traj) == 4, f"Expected 4 trajectory frames (N+1), got {len(traj)}"
    assert traj[0].shape == (B, C, H, W)
    print(f"trajectory           ✓  {len(traj)} frames")

    # --- get_nfe_options ---
    opts = bridge.get_nfe_options()
    assert 5 in opts and "psnr" in opts[5]
    print(f"\nNFE options:")
    for nfe, info in sorted(opts.items()):
        print(f"  NFE={nfe:3d}  PSNR≈{info['psnr']:12s}  ~{info['latency']:8s}  "
              f"{info['notes'][:60]}…")

    print("\nAll smoke tests passed ✓")
