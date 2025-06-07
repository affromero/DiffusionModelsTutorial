"""Score Distillation Sampling (SDS) Inpainting Example."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tyro
from torch import nn

from config import Config
from scripts.sampler import load_trained_model
from scripts.train import get_data_loaders
from src.diffusion.scheduler import DDIMScheduler, DDPMScheduler, PNDMScheduler
from src.models.utils import create_mask

torch.autograd.set_detect_anomaly(True)


def sds_loss(
    model: nn.Module,
    opt_image: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    digit_class: int,
    scheduler: DDIMScheduler | DDPMScheduler | PNDMScheduler,
    min_timestep: int = 20,
    max_timestep: int = 980,
    guidance_scale: float = 7.5,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute Score Distillation Sampling (SDS) loss.

    Mathematical Background:
    SDS loss = E_t,ε [w(t) * (ε_θ(x_t, t, c) - ε) * ∂x_t/∂x]

    Where:
    - x_t = √α̅_t * x + √(1-α̅_t) * ε (noised image)
    - ε_θ(x_t, t, c) is the model's noise prediction
    - w(t) is a weighting function
    - ∂x_t/∂x = √α̅_t (gradient of noised image w.r.t. clean image)

    Args:
        model: Trained diffusion model
        opt_image: Current optimized image (requires grad)
        gt_image: Ground truth image for unmasked regions
        mask: Binary mask (1=optimize, 0=preserve)
        digit_class: Target class for generation
        scheduler: Noise scheduler with precomputed constants
        min_timestep: Minimum timestep (avoid very noisy)
        max_timestep: Maximum timestep (avoid very clean)
        guidance_scale: Classifier-free guidance scale
        device: Device

    Returns:
        SDS loss tensor

    """
    batch_size = opt_image.shape[0]

    # Sample random timestep for this iteration
    valid_timesteps = torch.arange(min_timestep, max_timestep, device=device)
    t = valid_timesteps[torch.randint(0, len(valid_timesteps), (batch_size,))]

    # Sample noise
    noise = torch.randn_like(opt_image)

    # Add noise to current image: x_t = √α̅_t * x + √(1-α̅_t) * ε
    sqrt_alphas_cumprod_t = scheduler.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[
        t
    ].reshape(-1, 1, 1, 1)

    x_t = sqrt_alphas_cumprod_t * opt_image + sqrt_one_minus_alphas_cumprod_t * noise

    # Conditional prediction
    class_tensor = torch.full(
        (batch_size,), digit_class, device=device, dtype=torch.long
    )
    noise_pred_cond = model(x_t, t, class_tensor)

    # Unconditional prediction for CFG
    null_class = torch.zeros(batch_size, device=device, dtype=torch.long)
    noise_pred_uncond = model(x_t, t, null_class)

    # Apply classifier-free guidance
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_cond - noise_pred_uncond
    )

    # Compute SDS gradient direction
    # w(t) = α̅_t / (1 - α̅_t) - standard SDS weighting
    alphas_cumprod_t = scheduler.alphas_cumprod[t].reshape(-1, 1, 1, 1)
    w_t = alphas_cumprod_t / (1 - alphas_cumprod_t)

    # SDS gradient: w(t) * (ε_θ - ε) * ∂x_t/∂x
    # Since x_t = √α̅_t * x + √(1-α̅_t) * ε, we have ∂x_t/∂x = √α̅_t
    grad_direction = w_t * (noise_pred - noise) * sqrt_alphas_cumprod_t

    # Apply only to masked regions
    grad_direction = grad_direction * mask

    # SDS loss: we want the gradient to point in the direction that reduces the score
    # This is implemented as: grad_direction.detach() * opt_image
    # The .detach() prevents double gradients
    sds_loss = torch.mean(grad_direction.detach() * opt_image)

    # Add mask preservation loss
    mask_loss = torch.mean(((opt_image - gt_image) * (1 - mask)) ** 2)

    # Total loss
    return sds_loss + 100.0 * mask_loss  # Weight mask loss heavily


def sds_optimization(
    model: nn.Module,
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    digit_class: int,
    scheduler: DDIMScheduler | DDPMScheduler | PNDMScheduler,
    num_iterations: int = 500,
    learning_rate: float = 0.01,
    device: str = "cuda",
) -> torch.Tensor:
    """Run SDS optimization loop for inpainting.

    Args:
        model: Trained diffusion model
        gt_image: Ground truth image
        mask: Binary mask (1=inpaint, 0=preserve)
        digit_class: Target digit class
        scheduler: Noise scheduler
        num_iterations: Number of optimization steps
        learning_rate: Learning rate
        device: Device

    Returns:
        Optimized image

    """
    # Initialize optimizable image
    # Start with ground truth in unmasked regions, noise in masked regions
    noise_init = torch.randn_like(gt_image) * 0.5
    opt_image = mask * noise_init + (1 - mask) * gt_image
    opt_image = opt_image.detach().requires_grad_(True)  # noqa: FBT003

    # Setup optimizer
    optimizer = torch.optim.Adam([opt_image], lr=learning_rate)

    print(f"Starting SDS optimization for digit {digit_class}...")

    # turn off gradient calculation for the model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    for i in range(num_iterations):
        optimizer.zero_grad()

        # Compute SDS loss
        loss = sds_loss(
            model=model,
            opt_image=opt_image,
            gt_image=gt_image,
            mask=mask,
            digit_class=digit_class,
            scheduler=scheduler,
            device=device,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([opt_image], 1.0)

        # Optimizer step
        optimizer.step()

        # Force preservation of unmasked regions
        with torch.no_grad():
            opt_image.data = mask * opt_image.data.clamp(-1, 1) + (1 - mask) * gt_image

        if i % 100 == 0:
            print(f"  Iteration {i}: Loss = {loss.item():.4f}")

    return opt_image.clamp(-1, 1).detach()


# Example usage
def run_task(model_path: str = "outputs/models/best_model.pth") -> torch.Tensor:
    """Run SDS inpainting example."""
    config = Config.load_pkl()
    model, scheduler = load_trained_model(model_path, config)
    config.scheduler = DDIMScheduler()

    # Test image and mask
    _, val_loader = get_data_loaders(config)
    images = []
    labels = []
    for batch_images, batch_labels in val_loader:
        images.append(batch_images)
        labels.append(batch_labels)
    images_pt = torch.cat(images, dim=0)
    labels_pt = torch.cat(labels, dim=0)
    rand_idx = torch.randint(0, images_pt.size(0), (1,)).item()
    gt_image = images_pt[rand_idx][None].to(config.device)
    class_label = labels_pt[rand_idx].item()
    mask = create_mask(config.model.img_size, 10, (5, 5)).to(config.device)

    # Run SDS optimization
    result = sds_optimization(
        model=model,
        gt_image=gt_image,
        mask=mask,
        digit_class=class_label,
        scheduler=scheduler,
        num_iterations=500,
        learning_rate=0.01,
        device=config.device,
    )

    print("SDS optimization completed!")
    optimized_image = result

    # Mask the image for visualization purposes
    image_masked = gt_image * (1 - mask) + mask * (-1)

    # Define save path and ensure directory exists
    save_path = Path(config.sds_inpaint_dir) / "log.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the visualization
    save_sds_inpaint_visualization(
        gt_image=gt_image,
        mask=mask,
        image_masked=image_masked,
        optimized_image=optimized_image,
        save_path=str(save_path),
    )
    print(f"SDS inpainting visualization saved to {save_path}")

    return optimized_image


def save_sds_inpaint_visualization(
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    image_masked: torch.Tensor,
    optimized_image: torch.Tensor,
    save_path: str,
) -> None:
    """Save a 2x2 plot showing original, mask, masked, and optimized images."""
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(gt_image.cpu().squeeze().numpy(), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Mask")
    plt.imshow(mask.cpu().squeeze().numpy(), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Masked Image (Input to SDS)")
    display_masked = image_masked.cpu().squeeze().numpy()
    plt.imshow(display_masked, cmap="gray", vmin=-1, vmax=1)
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Optimized Image (SDS Result)")
    plt.imshow(optimized_image.detach().cpu().squeeze().numpy(), cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory and prevent display in non-interactive envs


if __name__ == "__main__":
    tyro.cli(run_task)
