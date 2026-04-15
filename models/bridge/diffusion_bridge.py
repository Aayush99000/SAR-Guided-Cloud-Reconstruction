"""Diffusion Bridge — forward process and training objective.

Architecture change from the old velocity-prediction formulation
----------------------------------------------------------------
The old bridge added Gaussian noise and trained the network to predict
the *velocity* (z₁ − z₀).  This file implements the deterministic
x₀-prediction formulation that matches the new BridgeNoiseSchedule:

  Forward  (deterministic):
    x_t = (1 − α_t) · x_clean  +  α_t · x_cloudy

  Training objective  (x₀-prediction):
    L = E_t [ λ_cloud · MSE(x̂₀ · M, x_clean · M)
              + λ_clear · MSE(x̂₀ · (1−M), x_clean · (1−M)) ]

    where M is the cloud binary mask (1 = cloudy pixel, 0 = clear pixel).

  Reverse ODE (inference):
    x_{t−s} = (1 − r) · x̂₀  +  r · x_t,   r = α_{t−s} / α_t

The network (SAROpticalUNet) predicts x̂₀ = f_θ(x_t, t, x_cloudy, SAR).
No VQ-GAN encoding is required — the bridge operates in pixel space.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .noise_schedule import BridgeNoiseSchedule


class DiffusionBridge(nn.Module):
    """Manages the forward process and training objective for pixel-space bridge.

    Args:
        model:           SAROpticalUNet (or any module matching its forward
                         signature: forward(x_t, t, x_cloudy_mean, sar,
                         cloud_mask) → x_hat_0).
        schedule_type:   "linear" | "sine" | "cosine"  (default "cosine").
        num_steps:       Training timestep count T  (default 1000).
        lambda_cloud:    Loss weight on cloud-masked pixels  (default 2.0).
        lambda_clear:    Loss weight on cloud-free pixels    (default 0.5).
        t_low:           Lower bound for training t samples  (default 0.02).
    """

    def __init__(
        self,
        model:          nn.Module,
        schedule_type:  str   = "cosine",
        num_steps:      int   = 1000,
        lambda_cloud:   float = 2.0,
        lambda_clear:   float = 0.5,
        t_low:          float = 0.02,
    ) -> None:
        super().__init__()
        self.model         = model
        self.schedule      = BridgeNoiseSchedule(num_steps=num_steps,
                                                 schedule_type=schedule_type)
        self.lambda_cloud  = lambda_cloud
        self.lambda_clear  = lambda_clear
        self.t_low         = t_low

    # ------------------------------------------------------------------
    # Forward process helper
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_clean:  torch.Tensor,
        x_cloudy: torch.Tensor,
        t:        torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t from the deterministic forward bridge.

        x_t = (1 − α_t) · x_clean  +  α_t · x_cloudy

        Args:
            x_clean:  Clean reference image  (B, C, H, W).
            x_cloudy: Cloudy input           (B, C, H, W).
            t:        Normalized time        (B,)  in [0, 1].

        Returns:
            x_t  (B, C, H, W).
        """
        return self.schedule.q_sample(x_clean, x_cloudy, t)

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------

    def training_loss(
        self,
        x_clean:       torch.Tensor,
        x_cloudy:      torch.Tensor,
        x_cloudy_mean: torch.Tensor,
        sar:           torch.Tensor,
        cloud_mask:    Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the x₀-prediction training loss.

        Samples a random t, constructs x_t via the forward bridge, runs the
        model to predict x̂₀, and applies a cloud-mask-weighted MSE.

        Args:
            x_clean:       Clean ground-truth image    (B, C, H, W)  in [0, 1].
            x_cloudy:      Cloudy input image          (B, C, H, W)  in [0, 1].
            x_cloudy_mean: Mean of cloudy observations (B, C, H, W)  — passed
                           to the model as additional conditioning.
            sar:           SAR image VV+VH             (B, 2, H, W)  in [0, 1].
            cloud_mask:    Binary cloud mask           (B, 1, H, W)  in {0, 1}.
                           1 = cloudy pixel, 0 = cloud-free pixel.
                           If None, all pixels are weighted equally.

        Returns:
            loss:    Scalar training loss.
            metrics: Dict with "loss", "cloud_mse", "clear_mse" values.
        """
        B = x_clean.shape[0]

        # Sample t uniformly from (t_low, 1.0)
        t = self.schedule.sample_t(B, device=x_clean.device, low=self.t_low)

        # Forward bridge: construct x_t
        x_t = self.schedule.q_sample(x_clean, x_cloudy, t)

        # Model prediction: x̂₀ = f_θ(x_t, t, x_cloudy_mean, SAR, mask)
        x_hat_0 = self.model(x_t, t, x_cloudy_mean, sar, cloud_mask)

        # Cloud-mask-weighted MSE
        sq_err = (x_hat_0 - x_clean) ** 2      # (B, C, H, W)

        if cloud_mask is not None:
            # cloud_mask: (B, 1, H, W)  →  broadcast over C
            cloud_pixels = cloud_mask                    # 1 = cloudy
            clear_pixels = 1.0 - cloud_mask             # 1 = clear

            n_cloud = cloud_pixels.sum().clamp(min=1.0)
            n_clear = clear_pixels.sum().clamp(min=1.0)

            cloud_mse = (sq_err * cloud_pixels).sum() / n_cloud
            clear_mse = (sq_err * clear_pixels).sum() / n_clear

            loss = self.lambda_cloud * cloud_mse + self.lambda_clear * clear_mse
        else:
            cloud_mse = sq_err.mean()
            clear_mse = sq_err.mean()
            loss      = cloud_mse

        metrics = {
            "loss":      loss.item(),
            "cloud_mse": cloud_mse.item(),
            "clear_mse": clear_mse.item(),
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_clean(
        self,
        x_t:           torch.Tensor,
        t:             torch.Tensor,
        x_cloudy_mean: torch.Tensor,
        sar:           torch.Tensor,
        cloud_mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run model forward to predict x̂₀ (no gradient)."""
        return self.model(x_t, t, x_cloudy_mean, sar, cloud_mask)

    @torch.no_grad()
    def sample(
        self,
        x_cloudy:      torch.Tensor,
        x_cloudy_mean: torch.Tensor,
        sar:           torch.Tensor,
        cloud_mask:    Optional[torch.Tensor] = None,
        num_steps:     int = 50,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Full reverse-ODE inference: x_T → x̂₀.

        Args:
            x_cloudy:      Cloudy starting image     (B, C, H, W).
            x_cloudy_mean: Mean cloudy conditioning  (B, C, H, W).
            sar:           SAR image                 (B, 2, H, W).
            cloud_mask:    Optional cloud mask       (B, 1, H, W).
            num_steps:     Number of denoising steps N.
            return_trajectory: If True, return list of all x_t states.

        Returns:
            Predicted clean image  (B, C, H, W), or list if return_trajectory.
        """
        timesteps = self.schedule.get_inference_steps(num_steps)  # (N+1,)
        device    = x_cloudy.device

        x_t       = x_cloudy.clone()       # start at t = 1 (fully cloudy)
        trajectory = [x_t] if return_trajectory else None

        for i in range(num_steps):
            t      = timesteps[i].to(device).expand(x_t.shape[0])
            t_prev = timesteps[i + 1].to(device).expand(x_t.shape[0])

            x_hat_0 = self.predict_clean(x_t, t, x_cloudy_mean, sar, cloud_mask)
            x_t     = self.schedule.reverse_step(x_t, x_hat_0, t, t_prev)

            if return_trajectory:
                trajectory.append(x_t)

        return trajectory if return_trajectory else x_t
