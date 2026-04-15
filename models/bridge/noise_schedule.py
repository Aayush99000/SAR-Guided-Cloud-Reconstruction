"""Bridge noise schedule for the OT-ODE deterministic cloud removal bridge.

Convention
----------
α goes from 0 (clean) → 1 (cloudy):

    x_t = (1 − α_t) · x_clean  +  α_t · x_cloudy       α_0 = 0,  α_T = 1

The process is a *deterministic* linear interpolation — no additive noise.

Reverse ODE (DDIM-style x₀-prediction)
---------------------------------------
Given a model estimate x̂₀ of the clean image at time t, the next state is:

    x_{t−s} = (1 − r) · x̂₀  +  r · x_t,    r = α_{t−s} / α_t

Substituting the forward formula shows this is exactly:

    x_{t−s} = (1 − α_{t−s}) · x̂₀  +  α_{t−s} · x_cloudy

i.e. the bridge re-evaluated at t−s with the model's prediction as the
clean anchor — no VQ-GAN encoding, no noise.

Schedule shapes  (u = t/T ∈ [0, 1])
--------------------------------------
 linear  :  α(u) = u                         — constant rate
 sine    :  α(u) = sin(π·u / 2)              — fast start, slow finish
 cosine  :  α(u) = 1 − cos(π·u / 2)         — slow start, fast finish  ← DB-CR

Derivative comparison at u = 0 / 0.5 / 1:
  linear :  1.00  / 1.00  / 1.00
  sine   :  1.57  / 1.11  / 0.00   (fastest at start)
  cosine :  0.00  / 1.11  / 1.57   (slowest at start → preferred for cloud removal)

References
----------
  Ebel et al., "DBER: Cross-Modal Cloud Removal with Optimal Transport", 2022.
  Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Supported schedule identifiers
# ---------------------------------------------------------------------------

_VALID = ("linear", "sine", "cosine")


# ---------------------------------------------------------------------------
# BridgeNoiseSchedule — primary API
# ---------------------------------------------------------------------------

class BridgeNoiseSchedule:
    """Deterministic OT-bridge noise schedule.

    All time inputs are accepted in two forms:
      · Continuous  float / float-tensor  in [0, 1]   (u = t / T already done)
      · Discrete    int  / integer-tensor in [0, T]   (normalized internally)

    Args:
        num_steps:     Number of discrete training timesteps T (default 1000).
        schedule_type: "linear" | "sine" | "cosine"  (default "sine").
    """

    def __init__(
        self,
        num_steps:     int = 1000,
        schedule_type: str = "sine",
    ) -> None:
        if schedule_type not in _VALID:
            raise ValueError(
                f"Unknown schedule '{schedule_type}'. Choose from: {_VALID}"
            )
        self.T             = num_steps
        self.schedule_type = schedule_type
        self._eps          = 1e-8   # floor for safe division in reverse_step

    # ------------------------------------------------------------------
    # Core: α(t)
    # ------------------------------------------------------------------

    def alpha(
        self,
        t: Union[float, int, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Evaluate the schedule at time t.

        Integer inputs (or integer-typed tensors) in [0, T] are
        normalized to [0, 1] by dividing by T.  Float inputs are
        used as-is and clamped to [0, 1].

        Returns the same type as the input.
        """
        if isinstance(t, torch.Tensor):
            return self._alpha_torch(t)
        if isinstance(t, np.ndarray):
            return self._alpha_numpy(t)
        if isinstance(t, (int, np.integer)):
            return self._alpha_scalar(float(t) / self.T)
        u = float(t)
        if u > 1.0 + 1e-6:         # discrete value passed as float (e.g. 500.0)
            u = u / self.T
        return self._alpha_scalar(u)

    def derivative(
        self,
        t: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """d(α)/d(u) — derivative w.r.t. normalized u = t/T.

        Useful for schedule visualization and analysis.
        """
        if isinstance(t, torch.Tensor):
            u = t.float().clamp(0.0, 1.0)
            if self.schedule_type == "linear":
                return torch.ones_like(u)
            elif self.schedule_type == "sine":
                return (math.pi / 2.0) * torch.cos(math.pi / 2.0 * u)
            else:  # cosine
                return (math.pi / 2.0) * torch.sin(math.pi / 2.0 * u)

        u = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        if self.schedule_type == "linear":
            return np.ones_like(u)
        elif self.schedule_type == "sine":
            return (math.pi / 2.0) * np.cos(math.pi / 2.0 * u)
        else:
            return (math.pi / 2.0) * np.sin(math.pi / 2.0 * u)

    # ------------------------------------------------------------------
    # Forward process (deterministic interpolation)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_clean:  torch.Tensor,
        x_cloudy: torch.Tensor,
        t:        torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t from the forward bridge.

        x_t = (1 − α_t) · x_clean  +  α_t · x_cloudy

        Args:
            x_clean:  Clean target image  (B, C, H, W).
            x_cloudy: Cloudy source image (B, C, H, W).
            t:        Normalized timesteps (B,) in [0, 1].

        Returns:
            x_t  (B, C, H, W) — deterministic interpolation.
        """
        a = self._alpha_torch(t).view(-1, 1, 1, 1)     # (B, 1, 1, 1)
        return (1.0 - a) * x_clean + a * x_cloudy

    # ------------------------------------------------------------------
    # Reverse step (deterministic ODE)
    # ------------------------------------------------------------------

    def reverse_step(
        self,
        x_t:     torch.Tensor,
        x_hat_0: torch.Tensor,
        t:       Union[torch.Tensor, float],
        t_prev:  Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """Single deterministic reverse step (DDIM-style).

        x_{t−s} = (1 − r) · x̂₀  +  r · x_t,    r = α_{t−s} / α_t

        Equivalent to re-evaluating the bridge at t−s with x̂₀ as the
        clean anchor:
            x_{t−s} = (1 − α_{t−s}) · x̂₀  +  α_{t−s} · x_cloudy

        Args:
            x_t:     Current noisy state  (B, C, H, W).
            x_hat_0: Model prediction of x_clean  (B, C, H, W).
            t:       Current time  — scalar or (B,) tensor in [0, 1].
            t_prev:  Previous time — scalar or (B,) tensor in [0, 1].
                     Must satisfy t_prev < t.

        Returns:
            x_{t−s}  (B, C, H, W).
        """
        def _prep(v):
            v = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
            v = self._alpha_torch(v.to(x_t.device))
            return v.view(-1, 1, 1, 1) if v.dim() > 0 else v.reshape(1, 1, 1, 1)

        alpha_t    = _prep(t)
        alpha_prev = _prep(t_prev)

        r = alpha_prev / (alpha_t + self._eps)          # r ∈ [0, 1)
        return (1.0 - r) * x_hat_0 + r * x_t

    # ------------------------------------------------------------------
    # Inference timestep grid
    # ------------------------------------------------------------------

    def get_inference_steps(
        self,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Uniformly spaced timestep grid for inference.

        Returns N+1 values descending from 1.0 to 0.0:
            [1.0,  1 − 1/N,  1 − 2/N,  …,  1/N,  0.0]

        Step size s = 1 / N  (= T/N in discrete units).

        The inference loop should iterate adjacent pairs (t, t_prev):
            for t, t_prev in zip(steps[:-1], steps[1:]):
                x_hat_0 = model(x_t, t, …)
                x_t     = schedule.reverse_step(x_t, x_hat_0, t, t_prev)

        Args:
            num_inference_steps: N, number of denoising function evaluations.

        Returns:
            Tensor of shape (N+1,).
        """
        return torch.linspace(1.0, 0.0, num_inference_steps + 1)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def sample_t(
        self,
        batch_size: int,
        device:     Union[str, torch.device] = "cpu",
        low:        float = 0.02,
        high:       float = 1.00,
    ) -> torch.Tensor:
        """Sample random normalized timesteps for a training batch.

        Draws uniformly from [low, high] to avoid the trivial endpoints
        (t = 0 is x_clean exactly; t = 1 is the observed x_cloudy).

        Args:
            batch_size: Number of samples.
            device:     Target device.
            low:        Lower bound  (default 0.02).
            high:       Upper bound  (default 1.00).

        Returns:
            (B,) float tensor in [low, high].
        """
        return torch.empty(batch_size, device=device).uniform_(low, high)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BridgeNoiseSchedule("
            f"schedule_type='{self.schedule_type}', "
            f"num_steps={self.T})"
        )

    # ------------------------------------------------------------------
    # Private: type-specific formula dispatch
    # ------------------------------------------------------------------

    def _alpha_scalar(self, u: float) -> float:
        u = max(0.0, min(1.0, u))
        if   self.schedule_type == "linear":
            return u
        elif self.schedule_type == "sine":
            return math.sin(math.pi * u / 2.0)
        else:   # cosine
            return 1.0 - math.cos(math.pi * u / 2.0)

    def _alpha_numpy(self, t: np.ndarray) -> np.ndarray:
        if np.issubdtype(t.dtype, np.integer):
            u = np.clip(t.astype(np.float64) / self.T, 0.0, 1.0)
        else:
            u = np.clip(t.astype(np.float64), 0.0, 1.0)
        if   self.schedule_type == "linear":
            return u
        elif self.schedule_type == "sine":
            return np.sin(np.pi * u / 2.0)
        else:
            return 1.0 - np.cos(np.pi * u / 2.0)

    def _alpha_torch(self, t: torch.Tensor) -> torch.Tensor:
        if not t.is_floating_point():
            u = t.float().div(self.T).clamp(0.0, 1.0)
        else:
            u = t.float().clamp(0.0, 1.0)
        if   self.schedule_type == "linear":
            return u
        elif self.schedule_type == "sine":
            return torch.sin(math.pi / 2.0 * u)
        else:   # cosine
            return 1.0 - torch.cos(math.pi / 2.0 * u)


# ---------------------------------------------------------------------------
# Schedule plot
# ---------------------------------------------------------------------------

def plot_schedules(
    output_path: Union[str, Path] = "noise_schedule_comparison.png",
    num_points:  int = 500,
    dpi:         int = 150,
) -> None:
    """Plot α(u) and dα/du for all three schedule types and save to file.

    Requires matplotlib.

    Args:
        output_path: Destination PNG file path.
        num_points:  Number of points for the curve.
        dpi:         Output image resolution.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    u = np.linspace(0.0, 1.0, num_points)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Bridge Noise Schedules\n"
        r"$\alpha_t = 0$ (clean) → $\alpha_t = 1$ (cloudy)",
        fontsize=13, fontweight="bold",
    )

    palette   = {"linear": "#2196F3", "sine": "#FF9800", "cosine": "#4CAF50"}
    labels    = {
        "linear": "Linear  —  $\\alpha(u) = u$",
        "sine"  : "Sine    —  $\\alpha(u) = \\sin(\\pi u / 2)$",
        "cosine": "Cosine  —  $\\alpha(u) = 1 - \\cos(\\pi u / 2)$",
    }
    descs     = {
        "linear": "constant rate",
        "sine"  : "fast start, slow finish",
        "cosine": "slow start, fast finish  (DB-CR default)",
    }

    ax_alpha, ax_deriv = axes

    for name in ("linear", "sine", "cosine"):
        sched = BridgeNoiseSchedule(schedule_type=name)
        a     = sched._alpha_numpy(u)
        da    = sched.derivative(u)      # returns np.ndarray here

        kw = dict(color=palette[name], linewidth=2.2)

        ax_alpha.plot(u, a, label=f"{labels[name]}\n({descs[name]})", **kw)
        ax_deriv.plot(u, da, **kw)

    # Reference lines
    ax_alpha.axline((0, 0), slope=1, color="gray", linewidth=0.8,
                    linestyle="--", alpha=0.5, label="reference $y=u$")

    # Formatting — alpha plot
    ax_alpha.set_xlabel("Normalized time  $u = t / T$", fontsize=11)
    ax_alpha.set_ylabel(r"$\alpha_t$", fontsize=12)
    ax_alpha.set_title(r"Schedule shape  $\alpha(u)$", fontsize=11)
    ax_alpha.set_xlim(0, 1);  ax_alpha.set_ylim(-0.02, 1.02)
    ax_alpha.legend(fontsize=8.5, loc="upper left")
    ax_alpha.grid(True, alpha=0.3)
    ax_alpha.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax_alpha.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax_alpha.tick_params(which="minor", length=2)

    # Derivative annotations at u = 0
    for name, color in palette.items():
        sched = BridgeNoiseSchedule(schedule_type=name)
        d0 = float(sched.derivative(np.array([0.0])))
        ax_deriv.annotate(
            f"  {d0:.2f}",
            xy=(0.0, d0), xytext=(0.04, d0),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    ax_deriv.set_xlabel("Normalized time  $u = t / T$", fontsize=11)
    ax_deriv.set_ylabel(r"$d\alpha / du$", fontsize=12)
    ax_deriv.set_title(r"Schedule velocity  $d\alpha(u)/du$", fontsize=11)
    ax_deriv.set_xlim(0, 1);  ax_deriv.set_ylim(-0.05, 1.75)
    ax_deriv.axhline(1.0, color="gray", linewidth=0.8,
                     linestyle="--", alpha=0.5, label="reference $y=1$")
    ax_deriv.legend(fontsize=8.5)
    ax_deriv.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Schedule plot saved → {out.resolve()}")


# ---------------------------------------------------------------------------
# Backward-compatible shims  (used by models/bridge/diffusion_bridge.py)
# ---------------------------------------------------------------------------
# The original API used:
#     from .noise_schedule import BaseAlphaSchedule, get_schedule
#     self.schedule = get_schedule("sine")
#     alpha, sigma = self.schedule.alpha_sigma(t)   # t ∈ [0, 1]
#
# These wrappers keep that code importable while the bridge is being migrated
# to the new BridgeNoiseSchedule API.  They will be removed once
# diffusion_bridge.py is updated.

class BaseAlphaSchedule:
    """Backward-compat base class.  New code should use BridgeNoiseSchedule."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Legacy stochastic sigma (always zero in the deterministic bridge)."""
        return torch.zeros_like(t)

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.alpha(t), self.sigma(t)


class _LinearLegacy(BaseAlphaSchedule):
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(0.0, 1.0)


class _SineLegacy(BaseAlphaSchedule):
    """Original squared-sine  (sin²(πt/2)) kept for checkpoint compatibility."""
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi / 2.0 * t).pow(2)


class _CosineLegacy(BaseAlphaSchedule):
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(math.pi / 2.0 * t).pow(2)


_LEGACY_REGISTRY = {
    "linear":  _LinearLegacy,
    "sine":    _SineLegacy,
    "cosine":  _CosineLegacy,
}


def get_schedule(name: str) -> BaseAlphaSchedule:
    """Backward-compat factory.  Returns a legacy BaseAlphaSchedule instance."""
    key = name.lower()
    if key not in _LEGACY_REGISTRY:
        raise ValueError(
            f"Unknown schedule '{name}'. Available: {list(_LEGACY_REGISTRY.keys())}"
        )
    return _LEGACY_REGISTRY[key]()


# Alias so old import  ``from .noise_schedule import SineAlphaSchedule``  works
SineAlphaSchedule   = _SineLegacy
LinearAlphaSchedule = _LinearLegacy
CosineAlphaSchedule = _CosineLegacy


# ---------------------------------------------------------------------------
# Smoke-test + plot
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("BridgeNoiseSchedule — smoke test")
    print("=" * 60)

    # --- 1. scalar / numpy / torch alpha values ---
    for stype in _VALID:
        sched = BridgeNoiseSchedule(num_steps=1000, schedule_type=stype)
        print(f"\n{sched}")

        # Python scalar (continuous)
        a0   = sched.alpha(0.0);    a05  = sched.alpha(0.5);  a1 = sched.alpha(1.0)
        # Python int (discrete)
        a500 = sched.alpha(500)     # should equal ~alpha(0.5)

        print(f"  α(0.0)   = {a0:.4f}  (expected 0)")
        print(f"  α(0.5)   = {a05:.4f}")
        print(f"  α(500)   = {a500:.4f}  (discrete 500/1000 ≈ α(0.5): {'✓' if abs(a05-a500)<1e-6 else '✗'})")
        print(f"  α(1.0)   = {a1:.4f}  (expected 1)")

        # numpy batch
        u_np  = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        a_np  = sched.alpha(u_np)
        assert a_np.shape == (5,), f"numpy output shape: {a_np.shape}"

        # torch batch (float)
        t_th  = torch.linspace(0.0, 1.0, 5)
        a_th  = sched.alpha(t_th)
        assert a_th.shape == (5,)

        # torch batch (integer)
        t_int = torch.arange(0, 1001, 250, dtype=torch.long)
        a_int = sched.alpha(t_int)
        assert a_int.shape == (5,)
        np.testing.assert_allclose(
            a_th.numpy(), a_int.numpy(), atol=1e-5,
            err_msg=f"{stype}: float vs integer tensor mismatch"
        )

        # Boundary conditions
        assert abs(float(sched.alpha(torch.tensor(0.0)))) < 1e-6, "α(0) ≠ 0"
        assert abs(float(sched.alpha(torch.tensor(1.0))) - 1.0) < 1e-6, "α(1) ≠ 1"

        print(f"  Boundary + dtype consistency  ✓")

    # --- 2. Forward process ---
    print("\n--- Forward process (q_sample) ---")
    B, C, H, W = 4, 4, 64, 64
    x_clean  = torch.ones(B, C, H, W)
    x_cloudy = torch.zeros(B, C, H, W)
    t        = torch.tensor([0.0, 0.25, 0.5, 1.0])

    sched = BridgeNoiseSchedule(schedule_type="cosine")
    x_t   = sched.q_sample(x_clean, x_cloudy, t)

    assert x_t.shape == (B, C, H, W)
    # At t=0: x_t should be x_clean = 1
    assert abs(x_t[0].mean().item() - 1.0) < 1e-5, "q_sample(t=0) ≠ x_clean"
    # At t=1: x_t should be x_cloudy = 0
    assert abs(x_t[3].mean().item() - 0.0) < 1e-5, "q_sample(t=1) ≠ x_cloudy"
    print(f"  t=[0, 0.25, 0.5, 1.0]  →  x_t means = {x_t.mean(dim=(1,2,3)).tolist()}")
    print(f"  Boundary conditions  ✓")

    # --- 3. Reverse step ---
    print("\n--- Reverse step ---")
    t_cur  = torch.full((B,), 0.5)
    t_prev = torch.full((B,), 0.25)
    x_hat_0 = torch.ones(B, C, H, W)     # model predicts all-clean

    x_t_input = sched.q_sample(x_clean, x_cloudy, t_cur)
    x_prev    = sched.reverse_step(x_t_input, x_hat_0, t_cur, t_prev)
    assert x_prev.shape == (B, C, H, W)

    # Last reverse step: t_prev=0 → result should be x_hat_0
    x_final = sched.reverse_step(x_t_input, x_hat_0,
                                  torch.full((B,), 0.5),
                                  torch.zeros(B))
    assert abs(x_final.mean().item() - 1.0) < 1e-5, "reverse_step(t_prev=0) ≠ x_hat_0"
    print(f"  reverse_step(t=0.5 → 0.25)  output mean = {x_prev.mean().item():.4f}  ✓")
    print(f"  reverse_step(t=0.5 → 0.00)  output mean = {x_final.mean().item():.4f}  (≈1.0)  ✓")

    # --- 4. Inference grid ---
    print("\n--- Inference timestep grid ---")
    N     = 10
    steps = sched.get_inference_steps(N)
    assert steps.shape == (N + 1,)
    assert abs(steps[0].item()  - 1.0) < 1e-6, "first step ≠ 1.0"
    assert abs(steps[-1].item() - 0.0) < 1e-6, "last step ≠ 0.0"
    assert all(steps[i] > steps[i+1] for i in range(N)), "steps not monotonically decreasing"
    print(f"  get_inference_steps(10) = {steps.tolist()}")
    print(f"  Monotone + boundary  ✓")

    # --- 5. Backward-compat shims ---
    print("\n--- Backward-compat shims ---")
    legacy = get_schedule("sine")
    a, s = legacy.alpha_sigma(torch.tensor([0.0, 0.5, 1.0]))
    assert s.sum().item() == 0.0, "legacy sigma should be zero"
    print(f"  get_schedule('sine') alpha = {a.tolist()}  sigma = {s.tolist()}  ✓")

    # --- 6. Plot ---
    print("\n--- Generating schedule comparison plot ---")
    plot_path = Path(__file__).parent.parent.parent / "noise_schedule_comparison.png"
    plot_schedules(output_path=plot_path)

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
