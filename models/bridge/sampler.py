"""ODE sampler for the diffusion bridge reverse process.

Supports 1–5 NFE (number of function evaluations) using either:
  - Euler method      (1 NFE = single Euler step, O(h))
  - Midpoint method   (2 NFE, O(h^2))
  - RK4               (4 NFE, O(h^4))
  - Adaptive RK45     (variable NFE via scipy/torchdiffeq)

At inference the reverse ODE integrates from t=1 (target distribution)
back to t=0 (source → reconstructed clear image).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch


OdeFn = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


class ODESampler:
    """Numerical ODE integrator for the probability-flow reverse process.

    Args:
        velocity_fn:   Callable v(z_t, t, cond) → velocity tensor.
        method:        Integration method: "euler" | "midpoint" | "rk4".
        num_steps:     NFE budget (1–5 for fast inference, or more for quality).
        t_start:       Starting time (default 1.0 = fully noised / clear latent).
        t_end:         Ending time   (default 0.0 = clean reconstruction).
    """

    _METHODS = ("euler", "midpoint", "rk4")

    def __init__(
        self,
        velocity_fn: OdeFn,
        method: str = "euler",
        num_steps: int = 5,
        t_start: float = 1.0,
        t_end: float = 0.0,
    ) -> None:
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}, got '{method}'")
        self.velocity_fn = velocity_fn
        self.method = method
        self.num_steps = num_steps
        self.t_start = t_start
        self.t_end = t_end

    # ------------------------------------------------------------------
    # Single-step integrators
    # ------------------------------------------------------------------

    def _euler_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        v = self.velocity_fn(z, t, cond)
        return z + dt * v

    def _midpoint_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        v1 = self.velocity_fn(z, t, cond)
        z_mid = z + 0.5 * dt * v1
        t_mid = t + 0.5 * dt
        v2 = self.velocity_fn(z_mid, t_mid, cond)
        return z + dt * v2

    def _rk4_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        k1 = self.velocity_fn(z, t, cond)
        k2 = self.velocity_fn(z + 0.5 * dt * k1, t + 0.5 * dt, cond)
        k3 = self.velocity_fn(z + 0.5 * dt * k2, t + 0.5 * dt, cond)
        k4 = self.velocity_fn(z + dt * k3, t + dt, cond)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------
    # Full reverse trajectory
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        z_init: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        """Integrate the reverse ODE from t_start → t_end.

        Args:
            z_init:            Initial latent at t=t_start, shape (B, D, H, W).
            cond:              Optional conditioning (SAR embedding, etc.).
            return_trajectory: If True, return all intermediate states.

        Returns:
            Final latent z_0 (or list of intermediate latents if return_trajectory).
        """
        step_fn = {
            "euler": self._euler_step,
            "midpoint": self._midpoint_step,
            "rk4": self._rk4_step,
        }[self.method]

        # Build time grid (descending: 1.0 → 0.0)
        ts = torch.linspace(
            self.t_start,
            self.t_end,
            self.num_steps + 1,
            device=z_init.device,
            dtype=z_init.dtype,
        )

        z = z_init
        trajectory = [z] if return_trajectory else None

        for i in range(self.num_steps):
            t_curr = ts[i].expand(z.shape[0])   # (B,)
            dt = ts[i + 1] - ts[i]              # negative (going backwards)
            z = step_fn(z, t_curr, dt, cond)
            if return_trajectory:
                trajectory.append(z)

        return trajectory if return_trajectory else z
