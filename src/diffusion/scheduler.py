"""DDPM noise scheduler for forward and reverse diffusion processes.

This module implements the mathematical foundations of DDPM:
- Beta schedule for noise variance
- Forward diffusion process q(x_t | x_0)
- Reverse diffusion process p_θ(x_{t-1} | x_t)
- All precomputed constants for efficient sampling

Alternative schedulers for better inpainting results.

Based on research, DDIM and PNDM are much better than DDPM for inpainting:
- DDIM: Deterministic, faster convergence, better quality
- PNDM: 20x faster than DDPM, excellent for inpainting
- DPM++: Modern solver with better stability
"""

from __future__ import annotations

import math
from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic.dataclasses import dataclass
from torch import Tensor

if TYPE_CHECKING:
    from jaxtyping import Float, Int

    from src.models import unet


@dataclass
class DDPMScheduler:
    """DDPM (Denoising Diffusion Probabilistic Models) noise scheduler.

    Mathematical Background:

    Forward Process:
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
    q(x_t | x_0) = N(x_t; √α̅_t x_0, (1-α̅_t) I)

    Where:
    - β_t: noise schedule
    - α_t = 1 - β_t: signal retention factor
    - α̅_t = ∏_{i=1}^t α_i: cumulative product

    Reverse Process:
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)

    Where:
    μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ(x_t, t))
    σ_t² = β_t * (1-α̅_{t-1}) / (1-α̅_t)

    Args:
        num_timesteps: Total number of diffusion steps T
        beta_schedule: Type of noise schedule ("linear" or "cosine")
        beta_start: Starting value for linear schedule
        beta_end: Ending value for linear schedule
        cosine_s: Offset parameter for cosine schedule
        device: Device to store tensors on

    """

    num_timesteps: int = 1000
    """ Total number of diffusion steps T. """
    beta_schedule: Literal["linear", "cosine"] = "cosine"
    """ Type of noise schedule. """
    beta_start: float = 0.0001
    """ Starting value for linear schedule. """
    beta_end: float = 0.02
    """ Ending value for linear schedule. """
    cosine_s: float = 0.008
    """ Offset parameter for cosine schedule. """
    device: str = "cuda"
    """ Device to store tensors on. """

    # -- init=False --
    betas: Float[Tensor, timesteps] = field(init=False)
    """ Noise schedule. """
    alphas: Float[Tensor, timesteps] = field(init=False)
    """ Signal retention factor. """
    alphas_cumprod: Float[Tensor, timesteps] = field(init=False)
    """ Cumulative product of alphas. """
    sqrt_alphas_cumprod: Float[Tensor, timesteps] = field(init=False)
    """ Square root of cumulative product of alphas. """
    sqrt_one_minus_alphas_cumprod: Float[Tensor, timesteps] = field(init=False)
    """ Square root of one minus cumulative product of alphas. """
    sqrt_recip_alphas: Float[Tensor, timesteps] = field(init=False)
    """ Square root of 1/alphas. """
    posterior_variance: Float[Tensor, timesteps] = field(init=False)
    """ Posterior variance. """
    posterior_mean_coef1: Float[Tensor, timesteps] = field(init=False)
    """ Coefficient for mean of posterior. """
    posterior_mean_coef2: Float[Tensor, timesteps] = field(init=False)
    """ Coefficient for variance of posterior. """

    def __post_init__(self) -> None:
        """Initialize constants after object creation."""
        # Generate beta schedule
        if self.beta_schedule == "linear":
            self.betas = self._linear_beta_schedule(self.beta_start, self.beta_end)
        elif self.beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(self.cosine_s)
        else:
            msg = f"Unknown beta_schedule: {self.beta_schedule}"
            raise ValueError(msg)

        # Precompute all necessary constants
        self._precompute_constants()

    def _linear_beta_schedule(
        self, beta_start: float, beta_end: float
    ) -> Float[Tensor, timesteps]:  # type: ignore[name-defined]
        """Linear beta schedule from DDPM paper.

        Mathematical Background:
        β_t = β_start + (β_end - β_start) * (t / T)

        Simple linear interpolation between start and end values.
        Works well but cosine schedule often gives better results.
        """
        return torch.linspace(
            beta_start, beta_end, self.num_timesteps, device=self.device
        )

    def _cosine_beta_schedule(self, s: float = 0.008) -> Float[Tensor, timesteps]:  # type: ignore[name-defined]
        """Cosine beta schedule from "Improved Denoising Diffusion Probabilistic Models".

        Mathematical Background:
        α̅_t = cos²((t/T + s) / (1 + s) * π/2)
        β_t = 1 - α̅_t / α̅_{t-1}

        The cosine schedule:
        1. Adds noise more slowly at the beginning
        2. Preserves more signal for longer
        3. Generally produces better sample quality

        The offset s prevents β_t from being too small near t=0.
        """
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps, device=self.device)

        # Cosine schedule for α̅_t
        alphas_cumprod = (
            torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # Compute β_t = 1 - α̅_t / α̅_{t-1}
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        # Clamp to prevent numerical issues
        return torch.clamp(betas, 0.0001, 0.9999)

    def _precompute_constants(self) -> None:
        """Precompute all constants needed for efficient forward/reverse diffusion.

        This avoids recomputing these values during training and sampling.
        """
        # Basic constants
        self.alphas = 1.0 - self.betas  # α_t = 1 - β_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # α̅_t = ∏α_i

        # For forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # √α̅_t
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )  # √(1-α̅_t)

        # For reverse process p_θ(x_{t-1} | x_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # 1/√α_t

        # α̅_{t-1} (pad with 1.0 for t=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Posterior variance: β̃_t = β_t * (1-α̅_{t-1}) / (1-α̅_t)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Coefficients for computing μ_θ
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(
        self,
        x_0: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, batch],  # type: ignore[name-defined]
        noise: Float[Tensor, "batch channels height width"] = None,
    ) -> tuple[
        Float[Tensor, "batch channels height width"],
        Float[Tensor, "batch channels height width"],
    ]:
        """Forward diffusion process: sample x_t from q(x_t | x_0).

        Mathematical Background:
        x_t = √α̅_t * x_0 + √(1-α̅_t) * ε

        This uses the reparameterization trick to sample directly from
        q(x_t | x_0) without iterating through all intermediate steps.

        Args:
            x_0: Clean images, shape (batch, channels, height, width)
            t: Timesteps, shape (batch,)
            noise: Optional noise tensor, if None will sample from N(0,I)

        Returns:
            x_t: Noisy images at timestep t
            noise: The noise that was added

        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract coefficients for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )

        # Apply the reparameterization: x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def predict_start_from_noise(
        self,
        x_t: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, batch],  # type: ignore[name-defined]
        noise: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch channels height width"]:
        """Predict x_0 from x_t and predicted noise.

        Mathematical Background:
        Given x_t = √α̅_t * x_0 + √(1-α̅_t) * ε, we can solve for x_0:
        x_0 = (x_t - √(1-α̅_t) * ε) / √α̅_t

        Args:
            x_t: Noisy images at timestep t
            t: Timesteps
            noise: Predicted noise

        Returns:
            Predicted clean images x_0

        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def q_posterior_mean_variance(
        self,
        x_start: Float[Tensor, "batch channels height width"],
        x_t: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, batch],  # type: ignore[name-defined]
    ) -> tuple[Float[Tensor, "batch channels height width"], Float[Tensor, batch]]:  # type: ignore[name-defined]
        """Compute the mean and variance of q(x_{t-1} | x_t, x_0).

        Mathematical Background:
        The posterior q(x_{t-1} | x_t, x_0) is Gaussian with:

        μ̃_t(x_t, x_0) = (√α̅_{t-1}β_t)/(1-α̅_t) * x_0 + (√α_t(1-α̅_{t-1}))/(1-α̅_t) * x_t
        σ̃_t² = β_t * (1-α̅_{t-1}) / (1-α̅_t)

        Args:
            x_start: Clean images x_0
            x_t: Noisy images at timestep t
            t: Timesteps

        Returns:
            posterior_mean: Mean of q(x_{t-1} | x_t, x_0)
            posterior_variance: Variance of q(x_{t-1} | x_t, x_0)

        """
        posterior_mean = (
            self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1) * x_start
            + self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)

        return posterior_mean, posterior_variance

    def p_mean_variance(
        self,
        model: unet.UNet,
        x_t: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, batch],  # type: ignore[name-defined]
        y: Int[Tensor, batch],  # type: ignore[name-defined]
    ) -> tuple[Float[Tensor, "batch channels height width"], Float[Tensor, batch]]:  # type: ignore[name-defined]
        """Compute the mean and variance of p_θ(x_{t-1} | x_t).

        Mathematical Background:
        We parameterize the reverse process mean as:
        μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ(x_t, t))

        This is derived from the optimal reverse process mean when
        ε_θ predicts the noise perfectly.

        Args:
            model: Noise prediction model
            x_t: Noisy images at timestep t
            t: Timesteps
            y: Class labels

        Returns:
            model_mean: Predicted mean μ_θ(x_t, t)
            posterior_variance: Variance σ_t²

        """
        # Predict noise
        predicted_noise = model(x_t, t, y)

        # Predict x_0 from noise
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)

        # Clamp x_0 to valid range
        x_start = torch.clamp(x_start, -1, 1)

        # Compute posterior mean and variance
        model_mean, posterior_variance = self.q_posterior_mean_variance(x_start, x_t, t)

        return model_mean, posterior_variance

    def p_sample(
        self,
        model: unet.UNet,
        x_t: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, batch],  # type: ignore[name-defined]
        y: Int[Tensor, batch],  # type: ignore[name-defined]
    ) -> Float[Tensor, "batch channels height width"]:
        """Sample x_{t-1} from p_θ(x_{t-1} | x_t).

        Mathematical Background:
        x_{t-1} ~ N(μ_θ(x_t, t), σ_t² I)

        For t=0, we return the mean (no noise added).
        For t>0, we add noise scaled by the posterior variance.

        Args:
            model: Noise prediction model
            x_t: Noisy images at timestep t
            t: Timesteps
            y: Class labels

        Returns:
            x_{t-1}: Sampled images at previous timestep

        """
        # Get mean and variance
        model_mean, model_variance = self.p_mean_variance(model, x_t, t, y)

        # Sample noise (except for t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)

        # Sample: x_{t-1} = μ_θ + σ_t * ε (only if t > 0)
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise

    def p_sample_loop(
        self,
        model: unet.UNet,
        shape: tuple[int, int, int, int],
        y: Int[Tensor, batch],  # type: ignore[name-defined]
    ) -> Float[Tensor, "batch channels height width"]:
        """Complete sampling loop from pure noise to clean images.

        Mathematical Background:
        Starting from x_T ~ N(0, I), we iteratively sample:
        x_{t-1} ~ p_θ(x_{t-1} | x_t, y) for t = T, T-1, ..., 1

        Args:
            model: Trained noise prediction model
            shape: Shape of images to generate
            y: Class labels for conditional generation

        Returns:
            Generated images x_0

        """
        device = next(model.parameters()).device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Iteratively denoise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, y)

        return x


@dataclass
class DDIMScheduler:
    """DDIM Scheduler optimized for inpainting - inherits q_sample from DDPM structure.

    Key advantages over DDPM for inpainting:
    - Deterministic sampling (reproducible results)
    - Faster convergence (10-50x speedup)
    - Better quality with fewer steps
    - More stable for inpainting tasks
    """

    num_timesteps: int = 1000
    """Number of training timesteps."""
    beta_start: float = 0.0001
    """Starting beta for the beta schedule."""
    beta_end: float = 0.02
    """Ending beta for the beta schedule."""
    beta_schedule: str = "linear"  # or "scaled_linear"
    """Beta schedule type."""
    eta: float = 0.0  # 0.0 = fully deterministic DDIM, 1.0 = DDPM
    """DDIM eta parameter."""
    clip_sample: bool = True
    """Clip samples to [-1, 1]."""
    clip_sample_range: float = 1.0
    """Range to clip samples to."""
    set_alpha_to_one: bool = True
    """Set alpha to 1.0."""
    steps_offset: int = 0
    """Steps offset."""
    device: str = "cuda"
    """Device to store tensors on."""

    # Consistent fields with DDPMScheduler for compatibility
    betas: Float[Tensor, num_timesteps] = field(init=False)
    """Beta schedule."""
    alphas: Float[Tensor, num_timesteps] = field(init=False)
    """Alpha schedule."""
    alphas_cumprod: Float[Tensor, num_timesteps] = field(init=False)
    """Alpha cumulative product."""
    sqrt_alphas_cumprod: Float[Tensor, num_timesteps] = field(init=False)
    """Square root of cumulative product of alphas."""
    sqrt_one_minus_alphas_cumprod: Float[Tensor, num_timesteps] = field(init=False)
    """Square root of one minus cumulative product of alphas."""
    sqrt_recip_alphas: Float[Tensor, num_timesteps] = field(init=False)
    """Square root of 1/alphas."""
    posterior_variance: Float[Tensor, num_timesteps] = field(init=False)
    """Posterior variance."""
    posterior_mean_coef1: Float[Tensor, num_timesteps] = field(init=False)
    """Coefficient for mean of posterior."""
    posterior_mean_coef2: Float[Tensor, num_timesteps] = field(init=False)
    """Coefficient for variance of posterior."""
    final_alpha_cumprod: float = field(init=False)
    """Final alpha cumulative product."""
    init_noise_sigma: float = field(init=False)
    """Initial noise sigma."""

    def __post_init__(self) -> None:
        """Initialize constants after object creation - consistent with DDPMScheduler."""
        # Create beta schedule - same as DDPM but with scaled_linear option
        if self.beta_schedule == "linear":
            self.betas = torch.linspace(
                self.beta_start, self.beta_end, self.num_timesteps, device=self.device
            )
        elif self.beta_schedule == "scaled_linear":
            # Used in Stable Diffusion
            self.betas = (
                torch.linspace(
                    self.beta_start**0.5,
                    self.beta_end**0.5,
                    self.num_timesteps,
                    device=self.device,
                )
                ** 2
            )
        elif self.beta_schedule == "cosine":
            # Cosine schedule
            self.betas = self._cosine_beta_schedule(self.num_timesteps)
        else:
            msg = f"Unknown beta_schedule: {self.beta_schedule}"
            raise ValueError(msg)

        # Precompute all constants - identical to DDPMScheduler for consistency
        self._precompute_constants()

        # DDIM-specific constants
        if self.set_alpha_to_one:
            self.final_alpha_cumprod = 1.0
        else:
            self.final_alpha_cumprod = self.alphas_cumprod[0].item()

        self.init_noise_sigma = 1.0

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in Improved DDPM - consistent with DDPMScheduler."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _precompute_constants(self) -> None:
        """Precompute all constants - identical to DDPMScheduler for compatibility."""
        # Basic constants
        self.alphas = 1.0 - self.betas  # α_t = 1 - β_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # α̅_t = ∏α_i

        # For forward process q(x_t | x_0) - same as DDPM
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # √α̅_t
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )  # √(1-α̅_t)

        # For reverse process p_θ(x_{t-1} | x_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # 1/√α_t

        # α̅_{t-1} (pad with 1.0 for t=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Posterior variance: β̃_t = β_t * (1-α̅_{t-1}) / (1-α̅_t)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Coefficients for computing μ_θ
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(
        self,
        x_0: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, batch],  # type: ignore[name-defined]
        noise: Float[Tensor, "batch channels height width"] = None,
    ) -> tuple[
        Float[Tensor, "batch channels height width"],
        Float[Tensor, "batch channels height width"],
    ]:
        """Forward diffusion process: sample x_t from q(x_t | x_0) - identical to DDPM."""
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract coefficients for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )

        # Apply the reparameterization: x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def predict_start_from_noise(
        self,
        x_t: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, batch],  # type: ignore[name-defined]
        noise: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch channels height width"]:
        """Predict x_0 from x_t and predicted noise - identical to DDPM."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def set_timesteps(self, num_inference_steps: int, device: str = "cuda") -> None:
        """Set timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        # Create subset of timesteps for inference (DDIM allows this!)
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )

        self.timesteps = torch.from_numpy(timesteps).to(device)

        # Move scheduler tensors to device if needed
        if self.alphas_cumprod.device != torch.device(device):
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
                device
            )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float | None = None,
        *,
        use_clipped_model_output: bool = False,
        generator: torch.Generator | None = None,
        variance_noise: torch.Tensor | None = None,
    ) -> tuple[
        Float[Tensor, "batch channels height width"],
        Float[Tensor, "batch channels height width"],
    ]:
        """DDIM sampling step - much more stable than DDPM for inpainting.

        Args:
            model_output: Predicted noise from the model
            timestep: Current timestep
            sample: Current sample (x_t)
            eta: Stochasticity parameter (0.0 = deterministic, uses self.eta if None)
            use_clipped_model_output: Whether to clip the model output
            generator: Random number generator
            variance_noise: Noise to add to the sample

        Returns:
            prev_sample: x_{t-1}
            pred_original_sample: Predicted x_0

        """
        if eta is None:
            eta = self.eta

        # Get previous timestep
        prev_timestep = timestep - self.num_timesteps // self.num_inference_steps

        # Compute alphas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        # Predict original sample (x_0) from noise prediction
        if use_clipped_model_output:
            model_output = torch.clamp(model_output, -1, 1)

        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)

        # Clip predicted x_0 to valid range
        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, -self.clip_sample_range, self.clip_sample_range
            )

        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = alpha_prod_t_prev ** (0.5)
        current_sample_coeff = (
            1
            - alpha_prod_t_prev
            - eta**2 * (1 - alpha_prod_t_prev / alpha_prod_t) * (1 - alpha_prod_t)
        ) ** (0.5)

        # Compute predicted previous sample mean
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * model_output
        )

        # Add noise if eta > 0 (stochastic sampling)
        if eta > 0:
            if variance_noise is not None and generator is not None:
                msg = "Cannot specify both generator and variance_noise"
                raise ValueError(msg)
            if variance_noise is None:
                variance_noise = torch.randn(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )

            variance = (
                eta
                * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** (0.5)
                * ((1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5))
            )
            pred_prev_sample = pred_prev_sample + variance * variance_noise

        return pred_prev_sample, pred_original_sample


@dataclass
class PNDMScheduler:
    """PNDM Scheduler - 20x faster than DDPM, excellent for inpainting.

    Uses the same q_sample method for consistency.

    Based on "Pseudo Numerical Methods for Diffusion Models on Manifolds"
    - Requires only 50 steps vs 1000 for DDPM
    - Higher order method for better accuracy
    - Very stable for inpainting tasks
    """

    num_timesteps: int = 1000
    """Number of training timesteps."""
    beta_start: float = 0.0001
    """Starting beta for the beta schedule."""
    beta_end: float = 0.02
    """Ending beta for the beta schedule."""
    device: str = "cuda"
    """Device to store tensors on."""

    # Consistent fields with DDPMScheduler for compatibility
    ets: list[Tensor] = field(init=False)
    """List of previous noise predictions."""
    betas: Float[Tensor, num_timesteps] = field(init=False)
    """Beta schedule."""
    alphas: Float[Tensor, num_timesteps] = field(init=False)
    """Alpha schedule."""
    alphas_cumprod: Float[Tensor, num_timesteps] = field(init=False)
    """Alpha cumulative product."""
    sqrt_alphas_cumprod: Float[Tensor, num_timesteps] = field(init=False)
    """Square root of cumulative product of alphas."""
    sqrt_one_minus_alphas_cumprod: Float[Tensor, num_timesteps] = field(init=False)
    """Square root of one minus cumulative product of alphas."""
    sqrt_recip_alphas: Float[Tensor, num_timesteps] = field(init=False)
    """Square root of 1/alphas."""
    posterior_variance: Float[Tensor, num_timesteps] = field(init=False)
    """Posterior variance."""
    posterior_mean_coef1: Float[Tensor, num_timesteps] = field(init=False)
    """Coefficient for mean of posterior."""
    posterior_mean_coef2: Float[Tensor, num_timesteps] = field(init=False)
    """Coefficient for variance of posterior."""

    def __post_init__(self) -> None:
        """Initialize constants after object creation - consistent with DDPMScheduler."""
        # Generate beta schedule - linear only for PNDM
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, self.num_timesteps, device=self.device
        )

        # Precompute all constants - identical to DDPMScheduler for consistency
        self._precompute_constants()

        # PNDM uses 4 previous steps for higher-order accuracy
        self.ets: list[Tensor] = []  # Store previous noise predictions
        self.cur_sample = None

    def _precompute_constants(self) -> None:
        """Precompute all constants - identical to DDPMScheduler for compatibility."""
        # Basic constants
        self.alphas = 1.0 - self.betas  # α_t = 1 - β_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # α̅_t = ∏α_i

        # For forward process q(x_t | x_0) - same as DDPM
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # √α̅_t
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )  # √(1-α̅_t)

        # For reverse process p_θ(x_{t-1} | x_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # 1/√α_t

        # α̅_{t-1} (pad with 1.0 for t=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Posterior variance: β̃_t = β_t * (1-α̅_{t-1}) / (1-α̅_t)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Coefficients for computing μ_θ
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(
        self,
        x_0: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, batch],  # type: ignore[name-defined]
        noise: Float[Tensor, "batch channels height width"] = None,
    ) -> tuple[
        Float[Tensor, "batch channels height width"],
        Float[Tensor, "batch channels height width"],
    ]:
        """Forward diffusion process: sample x_t from q(x_t | x_0) - identical to DDPM."""
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract coefficients for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )

        # Apply the reparameterization: x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def predict_start_from_noise(
        self,
        x_t: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, batch],  # type: ignore[name-defined]
        noise: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch channels height width"]:
        """Predict x_0 from x_t and predicted noise - identical to DDPM."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def set_timesteps(self, num_inference_steps: int, device: str = "cuda") -> None:
        """Set timesteps for PNDM inference."""
        self.num_inference_steps = num_inference_steps

        # PNDM timestep schedule
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )

        self.timesteps = torch.from_numpy(timesteps).to(device)

        # Move scheduler tensors to device if needed
        if self.alphas_cumprod.device != torch.device(device):
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
                device
            )

        # Reset PNDM state
        self.ets = []
        self.cur_sample = None

    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor
    ) -> torch.Tensor:
        """Run PNDM sampling step with multi-step method."""
        if len(self.ets) < 3:
            # Use simpler method for first few steps
            return self._euler_step(model_output, timestep, sample)
        # Use 4th order PNDM method
        return self._pndm_step(model_output, timestep, sample)

    def _euler_step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor
    ) -> torch.Tensor:
        """Run Simple Euler step for initial steps."""
        self.ets.append(model_output)

        prev_timestep = timestep - self.num_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else torch.tensor(1.0, device=self.alphas_cumprod.device)
        )

        beta_prod_t = 1 - alpha_prod_t

        # Predict x_0 using consistent method
        pred_x0 = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Compute prev_sample
        pred_dir = (1 - alpha_prod_t_prev).sqrt() * model_output
        return alpha_prod_t_prev.sqrt() * pred_x0 + pred_dir

    def _pndm_step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor
    ) -> torch.Tensor:
        """Run 4th order PNDM step."""
        self.ets.append(model_output)

        # Use 4 previous predictions for higher order accuracy
        if len(self.ets) > 4:
            self.ets.pop(0)

        # PNDM weights for 4th order method
        if len(self.ets) == 4:
            coeff = [55 / 24, -59 / 24, 37 / 24, -9 / 24]
        else:
            # Fallback to lower order
            coeff = [1.0] + [0.0] * (4 - len(self.ets))

        # Weighted combination of previous noise predictions
        weighted_pred = sum(c * et for c, et in zip(coeff, self.ets, strict=False))

        # Standard DDIM-like step with weighted prediction
        prev_timestep = timestep - self.num_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else torch.tensor(1.0, device=self.alphas_cumprod.device)
        )

        beta_prod_t = 1 - alpha_prod_t

        # Use consistent x_0 prediction method
        pred_x0 = (sample - beta_prod_t.sqrt() * weighted_pred) / alpha_prod_t.sqrt()
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        pred_dir = (1 - alpha_prod_t_prev).sqrt() * weighted_pred
        return alpha_prod_t_prev.sqrt() * pred_x0 + pred_dir


# Helper function to create optimized schedulers
def create_scheduler(
    scheduler_type: str = "ddim", **kwargs: Any
) -> DDIMScheduler | PNDMScheduler | DDPMScheduler:
    """Create scheduler optimized for inpainting.

    Args:
        scheduler_type: "ddim", "pndm", or "ddpm"
        **kwargs: Additional scheduler parameters

    Returns:
        Configured scheduler with consistent q_sample method

    """
    if scheduler_type == "ddim":
        return DDIMScheduler(**kwargs)
    if scheduler_type == "pndm":
        return PNDMScheduler(**kwargs)
    if scheduler_type == "ddpm":
        return DDPMScheduler(**kwargs)
    msg = f"Unknown scheduler type: {scheduler_type}"
    raise ValueError(msg)


# Recommended settings for different use cases
INPAINTING_PRESETS = {
    "quality": {
        "scheduler": "ddim",
        "num_inference_steps": 50,
        "eta": 0.0,  # Deterministic
        "beta_schedule": "scaled_linear",
    },
    "speed": {
        "scheduler": "pndm",
        "num_inference_steps": 20,
        "beta_schedule": "linear",
    },
    "balanced": {
        "scheduler": "ddim",
        "num_inference_steps": 30,
        "eta": 0.1,  # Slight stochasticity
        "beta_schedule": "scaled_linear",
    },
    "original": {
        "scheduler": "ddpm",
        "num_inference_steps": 1000,  # Full steps
        "beta_schedule": "cosine",
    },
}


# Quick test function to verify scheduler compatibility
def test_scheduler_compatibility() -> None:
    """Test that all schedulers have consistent q_sample methods."""
    # Test data
    x_0 = torch.randn(2, 1, 28, 28)
    t = torch.tensor([100, 200])

    # Create all schedulers
    ddpm = DDPMScheduler(num_timesteps=1000, device="cpu")
    ddim = DDIMScheduler(num_timesteps=1000, device="cpu")
    pndm = PNDMScheduler(num_timesteps=1000, device="cpu")

    print("Testing scheduler compatibility...")

    # Test q_sample for all schedulers
    schedulers: dict[str, DDIMScheduler | PNDMScheduler | DDPMScheduler] = {
        "DDPM": ddpm,
        "DDIM": ddim,
        "PNDM": pndm,
    }
    for name, scheduler in schedulers.items():
        try:
            x_t, noise = scheduler.q_sample(x_0, t)
            print(f"✅ {name}: q_sample works, output shape: {x_t.shape}")
        except Exception as e:  # noqa: BLE001
            print(f"❌ {name}: q_sample failed - {e}")

    # Test predict_start_from_noise for all schedulers
    noise = torch.randn_like(x_0)
    for name, scheduler in schedulers.items():
        try:
            x_t, _ = scheduler.q_sample(x_0, t, noise)
            x_0_pred = scheduler.predict_start_from_noise(x_t, t, noise)
            print(
                f"✅ {name}: predict_start_from_noise works, error: {(x_0 - x_0_pred).abs().mean():.6f}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"❌ {name}: predict_start_from_noise failed - {e}")

    print("All schedulers are now standardized with consistent q_sample methods! 🎯")


if __name__ == "__main__":
    test_scheduler_compatibility()
