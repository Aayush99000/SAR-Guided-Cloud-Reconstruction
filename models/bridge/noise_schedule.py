"""Noise schedules for the OT-guided diffusion bridge.

All schedules define alpha_t (signal coefficient) and sigma_t (noise coefficient)
such that the forward marginal is:

    q(z_t | z_0, z_1) = N(alpha_t * z_1 + (1 - alpha_t) * z_0, sigma_t^2 * I)

where z_0 is the source (cloudy latent) and z_1 is the target (clear latent).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


class BaseAlphaSchedule:
    """Abstract base for alpha schedules."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Default: sigma_t = sqrt(alpha_t * (1 - alpha_t)) * sigma_max."""
        a = self.alpha(t)
        return torch.sqrt(a * (1.0 - a)).clamp(min=1e-6)

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.alpha(t), self.sigma(t)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio alpha_t^2 / sigma_t^2."""
        a, s = self.alpha_sigma(t)
        return (a / s.clamp(min=1e-8)).pow(2)


class SineAlphaSchedule(BaseAlphaSchedule):
    """Sine-based schedule: alpha_t = sin^2(pi/2 * t).

    Provides smooth 0 → 1 interpolation with zero derivative at both ends,
    avoiding abrupt transitions at t=0 and t=1.
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi / 2.0 * t).pow(2)


class LinearAlphaSchedule(BaseAlphaSchedule):
    """Simple linear schedule: alpha_t = t."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(0.0, 1.0)


class CosineAlphaSchedule(BaseAlphaSchedule):
    """Cosine schedule (Nichol & Dhariwal 2021 variant): alpha_t = cos^2(pi/2 * t)."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(math.pi / 2.0 * t).pow(2)


_SCHEDULE_REGISTRY = {
    "sine": SineAlphaSchedule,
    "linear": LinearAlphaSchedule,
    "cosine": CosineAlphaSchedule,
}


def get_schedule(name: str) -> BaseAlphaSchedule:
    """Factory: retrieve a schedule by name string (from config)."""
    name = name.lower()
    if name not in _SCHEDULE_REGISTRY:
        raise ValueError(
            f"Unknown schedule '{name}'. Available: {list(_SCHEDULE_REGISTRY.keys())}"
        )
    return _SCHEDULE_REGISTRY[name]()
