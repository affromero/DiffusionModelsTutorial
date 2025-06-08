"""Sampling script for MNIST diffusion model.

This script loads a trained model and generates samples:
1. Load trained model from checkpoint
2. Generate samples for all digit classes
3. Create visualizations and save results

Usage:
    python scripts/sample.py --help
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tyro
from pydantic.dataclasses import dataclass

from config import Config
from src.data.dataset import denormalize_images
from src.diffusion.scheduler import DDIMScheduler, DDPMScheduler, PNDMScheduler
from src.models.unet import UNet
from src.models.utils import create_mask
from src.training.sampler import (
    DiffusionSampler,
    create_class_grid_visualization,
    visualize_samples,
)


def load_trained_model(
    model_path: str, config: Config
) -> tuple[UNet, DDPMScheduler | DDIMScheduler | PNDMScheduler]:
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")

    model = UNet(
        img_channels=config.model.img_channels,
        img_size=config.model.img_size,
        time_emb_dim=config.model.time_emb_dim,
        num_classes=config.model.num_classes,
        base_channels=config.model.base_channels,
        channel_mults=config.model.channel_mults,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    ).to(config.device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully!")
    print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Training loss: {checkpoint.get('loss', 'Unknown'):.4f}")

    return model, config.scheduler


@dataclass
class SamplingArgs:
    """Sampling arguments."""

    model_path: str = "outputs/models/best_model.pth"
    """Path to trained model checkpoint."""
    num_samples: int = 6
    """Number of samples per class."""
    specific_class: int | None = None
    """Generate samples for specific class only."""
    output_dir: str | None = None
    """Output directory (default: config.sample_dir)."""
    seed: int = 42
    """Random seed for sampling."""
    show_progress: bool = True
    """Show sampling progress bar."""
    interpolate: tuple[int, int] | None = None
    """Create interpolation between two classes."""
    inpaint: bool = False
    """Use inpainting."""
    guidance_scale: float = 1.0
    """Classifier-Free Guidance scale. If None, uses value from config.model.guidance_scale. 0.0 for unconditional, 1.0 for conditional (no CFG), >1.0 for stronger guidance."""

    def __post_init__(self) -> None:
        """Check for invalid combinations."""
        if self.inpaint and self.interpolate:
            msg = "Cannot use inpaint and interpolate at the same time."
            raise ValueError(msg)


def main(args: SamplingArgs, config: Config) -> None:
    """Run sampling pipeline."""
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.sample_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MNIST Diffusion Model Sampling")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Random Seed: {args.seed}")
    print(f"Output Directory: {output_dir}")

    # Load trained model
    model, scheduler = load_trained_model(args.model_path, config)

    # Create sampler
    sampler = DiffusionSampler(model, scheduler, config.device)

    # Set guidance_scale from config if not provided in args
    if args.guidance_scale != 1.0:
        print(f"Using Guidance Scale: {args.guidance_scale}")

    if args.interpolate:
        # Generate interpolation between two classes
        class1, class2 = args.interpolate
        print(
            f"\nGenerating interpolation between class {class1} and class {class2}..."
        )

        interpolated = sampler.interpolate_between_classes(
            class1=class1,
            class2=class2,
            num_steps=10,
            num_samples=1,
            img_size=config.model.img_size,
        )

        interpolated = denormalize_images(interpolated.squeeze(1).cpu())
        interpolated = torch.clamp(interpolated, 0, 1)

        fig, axes = plt.subplots(1, 10, figsize=(20, 2))
        for i in range(10):
            axes[i].imshow(interpolated[i, 0], cmap="gray")
            axes[i].axis("off")
            if i == 0:
                axes[i].set_title(f"Class {class1}")
            elif i == 9:
                axes[i].set_title(f"Class {class2}")

        plt.suptitle(
            f"Interpolation from Digit {class1} to Digit {class2}", fontsize=14
        )
        plt.tight_layout()

        # Save interpolation
        interp_path = output_dir / f"interpolation_{class1}_to_{class2}.png"
        plt.savefig(interp_path, dpi=150, bbox_inches="tight")
        print(f"Interpolation saved to {interp_path}")
        plt.show()

    elif args.specific_class is not None:
        # Generate samples for specific class
        print(
            f"\nGenerating {args.num_samples} samples for digit class {args.specific_class}..."
        )

        class_labels = torch.full(
            (args.num_samples,), args.specific_class, dtype=torch.long
        )
        if args.inpaint:
            mask = create_mask(config.model.img_size, 10, (5, 5)).to(config.device)
            image = sampler.sample(
                num_samples=1,
                class_labels=class_labels[:1],
                img_size=config.model.img_size,
                guidance_scale=args.guidance_scale,
            )
            image_masked = image * (1 - mask) + mask * (-1)
            image_masked = image_masked.repeat(args.num_samples, 1, 1, 1)
            samples = sampler.inpaint(
                image_masked=image_masked,
                mask=mask,
                digit_class=args.specific_class,
            )
            samples = torch.cat([image, mask, samples], dim=0)
        else:
            samples = sampler.sample(
                num_samples=args.num_samples,
                class_labels=class_labels,
                show_progress=args.show_progress,
                img_size=config.model.img_size,
                guidance_scale=args.guidance_scale,
            )

        # Visualize samples
        labels = [args.specific_class] * args.num_samples
        save_path = output_dir / f"class_{args.specific_class}_samples.png"

        visualize_samples(
            samples=samples,
            class_labels=labels,
            title=f"Generated Samples - Digit {args.specific_class}",
            save_path=str(save_path),
            samples_per_row=args.num_samples,
            is_inpaint=args.inpaint,
        )

    else:
        # Generate samples for all classes
        print(f"\nGenerating {args.num_samples} samples for each digit class...")

        save_path = output_dir / "all_classes_grid.png"
        create_class_grid_visualization(
            sampler=sampler,
            samples_per_class=args.num_samples,
            save_path=str(save_path),
            img_size=config.model.img_size,
            inpaint=args.inpaint,
            guidance_scale=args.guidance_scale,
        )

        # Also generate individual class samples
        print("\nGenerating individual class samples...")
        for digit in range(10):
            class_labels = torch.full((args.num_samples,), digit, dtype=torch.long)
            if args.inpaint:
                mask = create_mask(config.model.img_size, 10, (5, 5)).to(config.device)
                image = sampler.sample(
                    num_samples=1,
                    class_labels=class_labels[:1],
                    show_progress=False,
                    img_size=config.model.img_size,
                )
                image_masked = image * (1 - mask) + mask * (-1)
                image_masked = image_masked.repeat(args.num_samples, 1, 1, 1)
                samples = sampler.inpaint(
                    image_masked=image_masked,
                    mask=mask,
                    digit_class=digit,
                )
                samples = torch.cat([image, mask, samples], dim=0)
            else:
                samples = sampler.sample(
                    num_samples=args.num_samples,
                    class_labels=class_labels,
                    show_progress=False,
                    img_size=config.model.img_size,
                )

            labels = [digit] * args.num_samples
            save_path = output_dir / f"digit_{digit}_samples.png"
            visualize_samples(
                samples=samples,
                class_labels=labels,
                title=f"Generated Digit {digit} Samples",
                save_path=str(save_path),
                samples_per_row=args.num_samples,
                figsize=(12, 6),
                is_inpaint=args.inpaint,
            )

            print(f"  Digit {digit} samples saved")

    print("\n" + "=" * 60)
    print("Sampling completed successfully!")
    print(f"Results saved in: {output_dir}")
    print("=" * 60)


def run_sample(
    model_path: str = "outputs/models/best_model.pth",
    *,
    inpaint: bool = False,
    specific_class: int | None = None,
    num_samples: int = 6,
) -> None:
    """Run sampling pipeline."""
    config = Config.load_pkl()
    config.scheduler = DDIMScheduler()
    args = SamplingArgs(
        model_path=model_path,
        inpaint=inpaint,
        specific_class=specific_class,
        num_samples=num_samples,
    )
    main(args, config)


if __name__ == "__main__":
    tyro.cli(run_sample)
