"""Sampling module for generating images from trained diffusion models.

This module implements:
- DDPM sampling with mathematical foundations
- Efficient sampling loops
- Conditional generation for specific digit classes
- Image post-processing and visualization utilities
"""

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float, Int
from pydantic.dataclasses import dataclass
from torch import Tensor
from tqdm import tqdm

from src.data.dataset import denormalize_images, normalize_images
from src.diffusion.scheduler import DDIMScheduler, DDPMScheduler, PNDMScheduler
from src.models.unet import UNet
from src.models.utils import create_mask


@dataclass
class DiffusionSampler:
    """Sampler for generating images from trained diffusion models.

    Mathematical Background:
    The sampling process implements the reverse diffusion chain:

    x_T ~ N(0, I) (start from pure noise)
    x_{t-1} ~ p_θ(x_{t-1} | x_t, y) for t = T, T-1, ..., 1

    Where the reverse process is parameterized as:
    p_θ(x_{t-1} | x_t, y) = N(x_{t-1}; μ_θ(x_t, t, y), σ_t²I)

    The mean is computed as:
    μ_θ(x_t, t, y) = (1/√α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ(x_t, t, y))

    Args:
        model: Trained U-Net model for noise prediction
        scheduler: DDPM noise scheduler with precomputed constants
        device: Device to run sampling on

    """

    model: UNet
    """Trained U-Net model for noise prediction."""
    scheduler: DDPMScheduler | DDIMScheduler | PNDMScheduler
    """DDPM noise scheduler with precomputed constants."""
    device: str = "cuda"
    """Device to run sampling on."""

    def __post_init__(self) -> None:
        """Move model to device and set to evaluation mode."""
        self.model.to(self.device)
        # Set model to evaluation mode
        self.model.eval()

    @torch.no_grad()
    def inpaint(
        self,
        image_masked: Float[Tensor, "batch channels height width"],
        mask: Float[Tensor, "batch channels height width"],
        digit_class: int,
    ) -> Float[Tensor, "batch channels height width"]:
        """Implement inpainting using RePaint algorithm at pixel level.

        RePaint Algorithm (Lugmayr et al. 2022):
        1. Start with noise in masked regions, original pixels in unmasked regions.
        2. For each reverse diffusion step t from T-1 down to 0:
           a) Denoise the entire image x_t using the model to get x_{t-1}_generated.
           b) Noise the original known pixels (unmasked regions of image_masked) to step t-1 to get x_{t-1}_known_noised.
           c) Combine: x_{t-1} = mask * x_{t-1}_generated + (1-mask) * x_{t-1}_known_noised.
        3. This ensures perfect preservation of known pixels while generating coherent content.

        Args:
            image_masked: Input image with original known pixels in unmasked regions. [B, C, H, W]
            mask: Binary mask where 1=inpaint (generate), 0=preserve (known). [B, C, H, W] or [B, 1, H, W]
            digit_class: Target digit class for conditional generation.

        Returns:
            image_inpainted: Final inpainted image. [B, C, H, W]

        """
        self.model.eval()

        # Move to device and ensure correct types/shapes
        image_masked = image_masked.to(device=self.device, dtype=torch.float32)
        mask = mask.to(device=self.device, dtype=torch.float32)

        # Ensure mask has same channels as image (e.g., [B, 1, H, W] -> [B, C, H, W])
        if mask.shape[1] == 1 and image_masked.shape[1] > 1:
            mask = mask.repeat(1, image_masked.shape[1], 1, 1)

        batch_size = image_masked.shape[0]

        # Class conditioning: y should be [batch_size]
        y = torch.full((batch_size,), digit_class, device=self.device, dtype=torch.long)

        # Step 1: Initialize x_T.
        # Masked regions are filled with Gaussian noise.
        # Unmasked regions are initialized with the original pixel values from image_masked.
        initial_noise = torch.randn_like(image_masked)
        x_t = mask * initial_noise + (1 - mask) * image_masked  # x_T

        # Get reverse timesteps for inference (from T-1 down to 0)
        timesteps_loop = torch.linspace(
            self.scheduler.num_timesteps - 1,
            0,
            self.scheduler.num_timesteps,
            dtype=torch.long,
            device=self.device,
        )

        # Iterate through timesteps from T-1 down to 0
        for i, t_scalar_val in enumerate(timesteps_loop):
            t_current_scalar = t_scalar_val.item()
            # Current timestep as a tensor for batch processing: shape [batch_size]
            t_current_batch = torch.full(
                (batch_size,), t_current_scalar, device=self.device, dtype=torch.long
            )

            # Step 2a (part 1): Denoise the entire image x_t using the model to get predicted noise
            predicted_noise = self.model(x_t, t_current_batch, y)

            # Get DDPM scheduler parameters for the current timestep t_current_batch.
            # Reshape to [batch_size, 1, 1, 1] for broadcasting.
            alpha_t = self.scheduler.alphas[t_current_batch].view(-1, 1, 1, 1)
            alpha_bar_t = self.scheduler.alphas_cumprod[t_current_batch].view(
                -1, 1, 1, 1
            )
            beta_t = self.scheduler.betas[t_current_batch].view(-1, 1, 1, 1)

            # Compute the mean of the reverse distribution p(x_{t-1} | x_t)
            # x_mean = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * predicted_noise)
            sqrt_alpha_t_clamped = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
            sqrt_one_minus_alpha_bar_t_clamped = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_t, min=1e-8)
            )

            x_mean = (1.0 / sqrt_alpha_t_clamped) * (
                x_t - (beta_t / sqrt_one_minus_alpha_bar_t_clamped) * predicted_noise
            )

            # If not the last timestep (i.e., t_current_scalar > 0, so we are calculating x_{t-1})
            if t_current_scalar > 0:
                # Determine the next timestep for RePaint logic (which is t-1 for the current t)
                t_next_scalar = timesteps_loop[
                    i + 1
                ].item()  # This will be t_current_scalar - 1
                t_next_batch = torch.full(
                    (batch_size,), t_next_scalar, device=self.device, dtype=torch.long
                )

                # Calculate posterior variance (beta_tilde_t) for stochastic sampling of x_{t-1}
                # variance = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
                alpha_bar_next_for_variance = self.scheduler.alphas_cumprod[
                    t_next_batch
                ].view(-1, 1, 1, 1)
                variance = (
                    beta_t
                    * (1.0 - alpha_bar_next_for_variance)
                    / torch.clamp(1.0 - alpha_bar_t, min=1e-20)
                )
                variance = torch.clamp(variance, min=1e-20)  # Ensure non-negative

                # Step 2a (part 2): Sample x_{t-1}_generated from p(x_{t-1} | x_t)
                noise_sample = torch.randn_like(x_t)
                # Add noise only if variance is positive for any item in batch
                x_denoised_generated = (
                    x_mean
                    + torch.sqrt(variance) * noise_sample * (variance > 1e-8).float()
                )

                # Step 2b: RePaint - Prepare the known region (unmasked part) for step t-1.
                # Noise the original unmasked data (from image_masked) to the level of t_next_scalar (t-1).
                if (
                    t_next_scalar > 0
                ):  # If t-1 > 0 (i.e., we are not noising to x_0 level)
                    alpha_bar_next_gt = self.scheduler.alphas_cumprod[
                        t_next_batch
                    ].view(-1, 1, 1, 1)
                    sqrt_alpha_bar_next_gt_clamped = torch.sqrt(
                        torch.clamp(alpha_bar_next_gt, min=1e-8)
                    )
                    sqrt_one_minus_alpha_bar_next_gt_clamped = torch.sqrt(
                        torch.clamp(1.0 - alpha_bar_next_gt, min=1e-8)
                    )

                    noise_gt = torch.randn_like(
                        image_masked
                    )  # Fresh noise for ground truth noising
                    image_known_noised = (
                        sqrt_alpha_bar_next_gt_clamped
                        * image_masked  # image_masked holds original data
                        + sqrt_one_minus_alpha_bar_next_gt_clamped * noise_gt
                    ).clamp(-1, 1)
                else:  # If t_next_scalar is 0 (i.e., t-1 = 0), the known part is the original data itself.
                    image_known_noised = image_masked

                # Step 2c: Combine generated content (masked) and noised known content (unmasked).
                # This forms x_{t-1}, which becomes x_t for the next iteration.
                x_t = mask * x_denoised_generated + (1 - mask) * image_known_noised

            else:  # This is the last step (t_current_scalar == 0), we are calculating x_0.
                # The model's prediction x_mean is the final denoised image for masked parts.
                # Unmasked parts should remain the original image_masked pixels.
                # No more stochastic noise is added as variance is effectively zero (or not used).
                x_t = (
                    mask * x_mean + (1 - mask) * image_masked
                )  # Correctly combine for the final result

        return x_t

    @torch.no_grad()
    def sample(
        self,
        *,
        num_samples: int,
        class_labels: Int[Tensor, "batch"] | None = None,  # type: ignore[name-defined]
        guidance_scale: float | None = None,
        return_intermediates: bool = False,
        show_progress: bool = True,
        img_size: tuple[int, int],
    ) -> Float[Tensor, "batch channels height width"]:
        """Generate samples using DDPM reverse process.

        Supports Classifier-Free Guidance (CFG) if `guidance_scale` is provided
        and not equal to 1.0 (or 0.0 for purely unconditional).
        A `guidance_scale` of 0.0 means purely unconditional samples.
        A `guidance_scale` of 1.0 means purely conditional samples (no CFG effect).
        Values > 1.0 enhance class conditioning.

        Mathematical Background:
        Starting from x_T ~ N(0, I), we iteratively sample:

        For t = T, T-1, ..., 1:
            1. Predict noise: ε_θ(x_t, t, y)
            2. Compute mean: μ_θ = (1/√α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ)
            3. Sample: x_{t-1} ~ N(μ_θ, σ_t²I) if t > 0, else x_0 = μ_θ

        Args:
            num_samples: Number of samples to generate
            class_labels: Class labels for conditional generation (optional)
            return_intermediates: Whether to return intermediate steps
            show_progress: Whether to show progress bar
            img_size: Image size (height, width)
            intermediates_every: How often to store intermediate steps
            guidance_scale: Classifier-Free Guidance scale.

        Returns:
            samples: Generated images, shape (num_samples, 1, H, W)
            intermediates: List of intermediate steps (if return_intermediates=True)

        """
        # Initialize from pure noise
        img_shape = (num_samples, 1, *img_size)
        x = torch.randn(img_shape, device=self.device)

        # Handle class labels
        if class_labels is None:
            # Generate random class labels if not provided
            class_labels = torch.randint(0, 10, (num_samples,), device=self.device)
        else:
            class_labels = class_labels.to(self.device)

        # Store intermediates if requested
        if return_intermediates:
            intermediates: list[Float[Tensor, "batch channels height width"]] = []

        # Reverse diffusion process
        timesteps = reversed(range(self.scheduler.num_timesteps))
        if show_progress:
            timesteps = tqdm(
                timesteps, desc="Sampling", total=self.scheduler.num_timesteps
            )
        intermediates_every = self.scheduler.num_timesteps // 10
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            t_tensor = torch.full(
                (num_samples,), t, device=self.device, dtype=torch.long
            )

            # Predict noise
            if (
                guidance_scale is not None
                and guidance_scale != 1.0
                and class_labels is not None
            ):
                # Classifier-Free Guidance
                # Note: self.model's forward now accepts y=None for unconditional pass
                pred_noise_cond = self.model(x, t_tensor, class_labels)
                pred_noise_uncond = self.model(x, t_tensor, y=None)
                predicted_noise = pred_noise_uncond + guidance_scale * (
                    pred_noise_cond - pred_noise_uncond
                )
            elif guidance_scale == 0.0:  # Purely unconditional
                predicted_noise = self.model(x, t_tensor, y=None)
            else:  # Conditional (guidance_scale is 1.0 or None) or if class_labels were None initially
                # If class_labels is None here, self.model(..., y=None) handles it as unconditional.
                predicted_noise = self.model(x, t_tensor, class_labels)

            # Compute reverse process parameters
            self.scheduler.alphas[t]
            beta_t = self.scheduler.betas[t]
            sqrt_one_minus_alpha_cumprod_t = (
                self.scheduler.sqrt_one_minus_alphas_cumprod[t]
            )
            sqrt_recip_alpha_t = self.scheduler.sqrt_recip_alphas[t]

            # Compute mean: μ_θ = (1/√α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ)
            pred_x0 = sqrt_recip_alpha_t * (
                x - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
            )

            # Clamp predicted x_0 to valid range
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            if t > 0:
                # Add noise for stochastic sampling
                posterior_variance = self.scheduler.posterior_variance[t]
                noise = torch.randn_like(x)
                x = pred_x0 + torch.sqrt(posterior_variance) * noise
            else:
                # Final step: no noise added
                x = pred_x0

            # Store intermediate if requested
            if (
                return_intermediates and i % intermediates_every == 0
            ):  # Store every intermediates_every steps
                intermediates.append(x.clone().cpu())

        if return_intermediates:
            return x, intermediates
        return x

    def sample_class_grid(
        self,
        *,
        samples_per_class: int = 8,
        classes: list[int] | None = None,
        show_progress: bool = True,
        img_size: tuple[int, int],
        inpaint: bool = False,
        guidance_scale: float | None = None,
    ) -> tuple[Float[Tensor, "batch channels height width"], list[int]]:
        """Generate a grid of samples for each class.

        Args:
            samples_per_class: Number of samples per class
            classes: List of classes to generate (default: all 10 digits)
            show_progress: Whether to show progress bar
            img_size: Image size (height, width)
            inpaint: Use inpainting for generating samples
            guidance_scale: Classifier-Free Guidance scale.

        Returns:
            all_samples: Generated samples, shape (total_samples, 1, H, W)
            class_labels: Corresponding class labels

        """
        if classes is None:
            classes = list(range(10))  # All MNIST digits

        all_samples = []
        all_labels = []

        for class_idx in classes:
            # Create class labels
            class_labels = torch.full((samples_per_class,), class_idx, dtype=torch.long)

            # Generate samples for this class
            if inpaint:
                mask = create_mask(img_size, 10, (5, 5)).to(self.device)
                image = self.sample(
                    num_samples=1,
                    class_labels=class_labels[:1],
                    show_progress=show_progress,
                    img_size=img_size,
                    guidance_scale=guidance_scale,
                )
                image_masked = image * (1 - mask) + mask * (-1)
                image_masked = image_masked.repeat(samples_per_class, 1, 1, 1)
                samples = self.inpaint(
                    image_masked=image_masked,
                    mask=mask,
                    digit_class=class_idx,
                )
                samples = torch.cat([image, normalize_images(mask), samples], dim=0)
            else:
                samples = self.sample(
                    num_samples=samples_per_class,
                    class_labels=class_labels,
                    show_progress=show_progress,
                    img_size=img_size,
                    guidance_scale=guidance_scale,
                )

            all_samples.append(samples)
            all_labels.extend([class_idx] * samples_per_class)

        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)

        return all_samples, all_labels

    def interpolate_between_classes(
        self,
        class1: int,
        class2: int,
        num_steps: int = 10,
        num_samples: int = 1,
        *,
        img_size: tuple[int, int],
    ) -> Float[Tensor, "batch channels height width"]:
        """Generate interpolation between two digit classes.

        Mathematical Background:
        We interpolate in the noise space and class embedding space:
        - Start with noise for class1, end with noise for class2
        - Interpolate class embeddings linearly

        Args:
            class1: First digit class
            class2: Second digit class
            num_steps: Number of interpolation steps
            num_samples: Number of parallel interpolations
            img_size: Image size (height, width)

        Returns:
            interpolated_samples: Shape (num_steps, num_samples, 1, H, W)

        """
        # Generate two sets of initial noise
        img_shape = (num_samples, 1, *img_size)
        noise1 = torch.randn(img_shape, device=self.device)
        noise2 = torch.randn(img_shape, device=self.device)

        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps, device=self.device)

        interpolated_samples = []

        for alpha in tqdm(alphas, desc="Interpolating"):
            # Interpolate initial noise
            x = (1 - alpha) * noise1 + alpha * noise2

            # Interpolate class labels (will be rounded to nearest integer)
            interp_class = int((1 - alpha) * class1 + alpha * class2)
            class_labels = torch.full(
                (num_samples,), interp_class, device=self.device, dtype=torch.long
            )

            # Run reverse diffusion
            for t in reversed(range(self.scheduler.num_timesteps)):
                t_tensor = torch.full(
                    (num_samples,), t, device=self.device, dtype=torch.long
                )
                predicted_noise = self.model(x, t_tensor, class_labels)

                # Standard reverse step
                self.scheduler.alphas[t]
                beta_t = self.scheduler.betas[t]
                sqrt_one_minus_alpha_cumprod_t = (
                    self.scheduler.sqrt_one_minus_alphas_cumprod[t]
                )
                sqrt_recip_alpha_t = self.scheduler.sqrt_recip_alphas[t]

                pred_x0 = sqrt_recip_alpha_t * (
                    x - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
                )
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                if t > 0:
                    posterior_variance = self.scheduler.posterior_variance[t]
                    noise = torch.randn_like(x)
                    x = pred_x0 + torch.sqrt(posterior_variance) * noise
                else:
                    x = pred_x0

            interpolated_samples.append(x)

        return torch.stack(interpolated_samples, dim=0)


def visualize_samples(
    samples: Float[Tensor, "batch channels height width"],
    class_labels: list[int] | None = None,
    title: str = "Generated Samples",
    save_path: str | None = None,
    figsize: tuple[int, int] = (15, 8),
    samples_per_row: int = 8,
    *,
    is_inpaint: bool = False,
) -> None:
    """Visualize generated samples in a grid.

    Args:
        samples: Generated samples, shape (N, 1, H, W)
        class_labels: Class labels for each sample
        title: Title for the plot
        save_path: Path to save the figure (optional)
        figsize: Figure size
        samples_per_row: Number of samples per row
        is_inpaint: Whether the samples are inpainted

    """
    # Denormalize images for visualization
    samples = denormalize_images(samples.cpu())
    samples = torch.clamp(samples, 0, 1)

    num_samples = len(samples) - (2 if is_inpaint else 0)
    num_rows = (num_samples + samples_per_row - 1) // samples_per_row

    fig, axes = plt.subplots(num_rows, samples_per_row, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        row = i // samples_per_row
        col = i % samples_per_row

        ax = axes[row, col]
        ax.imshow(samples[i, 0], cmap="gray")
        ax.axis("off")

        if class_labels is not None:
            ax.set_title(f"Class {class_labels[i]}")

    # Hide empty subplots
    for i in range(num_samples, num_rows * samples_per_row):
        row = i // samples_per_row
        col = i % samples_per_row
        axes[row, col].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def create_class_grid_visualization(
    sampler: DiffusionSampler,
    *,
    samples_per_class: int = 4,
    save_path: str | None = None,
    img_size: tuple[int, int],
    inpaint: bool = False,
    guidance_scale: float | None = None,
) -> None:
    """Create and visualize a grid of samples for all digit classes.

    Args:
        sampler: Configured diffusion sampler
        samples_per_class: Number of samples per digit class
        save_path: Path to save the visualization
        img_size: Image size
        inpaint: Use inpainting for generating samples
        guidance_scale: Classifier-Free Guidance scale. If None, uses value from config.model.guidance_scale. 0.0 for unconditional, 1.0 for conditional (no CFG), >1.0 for stronger guidance.

    """
    print("Generating samples for all digit classes...")

    # Generate samples for all classes
    samples, labels = sampler.sample_class_grid(
        samples_per_class=samples_per_class,
        show_progress=True,
        img_size=img_size,
        inpaint=inpaint,
        guidance_scale=guidance_scale,
    )
    if inpaint:
        samples_per_class += 2
    # Create visualization
    fig, axes = plt.subplots(
        10,
        samples_per_class,
        figsize=(
            samples_per_class * 2,
            20,
        ),
    )
    total_classes = samples.shape[0] // samples_per_class
    # Denormalize samples
    # samples = denormalize_images(samples.cpu())
    samples = torch.clamp(samples, 0, 1)

    for class_idx in range(total_classes):
        class_samples = samples[
            class_idx * samples_per_class : (class_idx + 1) * samples_per_class
        ]

        for sample_idx in range(samples_per_class):
            ax = axes[class_idx, sample_idx]
            ax.imshow(class_samples[sample_idx, 0].cpu(), cmap="gray")
            ax.axis("off")

            if sample_idx == 0:
                ax.set_ylabel(
                    f"Digit {class_idx}", fontsize=14, rotation=0, labelpad=20
                )

    plt.suptitle("Generated MNIST Samples by Class", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Class grid saved to {save_path}")

    plt.show()
