"""Latent Diffusion Bridge (OT-ODE) forward and reverse process.

The bridge is defined between two latent distributions:
  - z_0 ~ p_src  (cloudy image latent, from VQ-GAN encoder)
  - z_1 ~ p_tgt  (clear image latent)

Forward process (Schrödinger Bridge / OT interpolation):
    z_t = (1 - alpha_t) * z_0  +  alpha_t * z_1  +  sigma_t * eps
    eps ~ N(0, I)

Reverse ODE (probability flow):
    dz/dt = v_theta(z_t, t, cond) + d(log alpha_t)/dt * (z_t - z_0)

A neural velocity field v_theta (the U-Net backbone) is trained to predict
the *straight* velocity (z_1 - z_0) at each time step.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .noise_schedule import BaseAlphaSchedule, get_schedule


class DiffusionBridge(nn.Module):
    """Manages the forward noising process and training objective.

    The reverse process (sampling) is handled by :class:`ODESampler`.

    Args:
        velocity_net:    The U-Net / backbone that predicts v(z_t, t, cond).
        schedule_name:   One of "sine", "linear", "cosine".
        diffusion_steps: Number of discrete time steps T used during training.
        ot_reg:          Optional OT regularisation weight (currently a loss scale).
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        schedule_name: str = "sine",
        diffusion_steps: int = 1000,
        ot_reg: float = 0.01,
    ) -> None:
        super().__init__()
        self.velocity_net = velocity_net
        self.schedule: BaseAlphaSchedule = get_schedule(schedule_name)
        self.T = diffusion_steps
        self.ot_reg = ot_reg

    # ------------------------------------------------------------------
    # Forward (noising) process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample z_t given endpoints z_0 and z_1.

        Args:
            z0: Source latent (B, D, H, W).
            z1: Target latent (B, D, H, W).
            t:  Time values in [0, 1] of shape (B,).

        Returns:
            z_t:  Noised latent.
            eps:  The noise sample used (for loss computation).
        """
        alpha, sigma = self.schedule.alpha_sigma(t)
        alpha = alpha.view(-1, 1, 1, 1)
        sigma = sigma.view(-1, 1, 1, 1)

        eps = torch.randn_like(z0)
        z_t = (1.0 - alpha) * z0 + alpha * z1 + sigma * eps
        return z_t, eps

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------

    def training_loss(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the bridge training loss.

        Uses a *flow-matching* / velocity-prediction objective:
            L = E_t,z_t [ ||v_theta(z_t, t, cond) - (z_1 - z_0)||^2 ]

        An optional OT regularisation term penalises the quadratic transport
        cost E[||z_1 - z_0||^2] to encourage straight trajectories.

        Args:
            z0:   Source latents (B, D, H, W).
            z1:   Target latents (B, D, H, W).
            cond: Optional conditioning tensor (e.g. SAR embedding).

        Returns:
            loss:    Scalar loss tensor.
            metrics: Dict with sub-loss values for logging.
        """
        B = z0.shape[0]
        # Sample t uniformly in (0, 1)
        t = torch.rand(B, device=z0.device)

        z_t, _ = self.q_sample(z0, z1, t)
        v_target = z1 - z0                                # (B, D, H, W)
        v_pred = self.velocity_net(z_t, t, cond)         # (B, D, H, W)

        flow_loss = ((v_pred - v_target) ** 2).mean()

        ot_loss = (v_target ** 2).mean() * self.ot_reg   # penalise long bridges

        loss = flow_loss + ot_loss
        return loss, {"flow_loss": flow_loss.item(), "ot_loss": ot_loss.item()}

    # ------------------------------------------------------------------
    # Convenience: score / velocity at inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the velocity network (no grad)."""
        return self.velocity_net(z_t, t, cond)
