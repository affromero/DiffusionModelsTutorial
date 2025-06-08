"""Training script for MNIST diffusion model.

This script demonstrates the complete training pipeline:
1. Load and preprocess MNIST data
2. Initialize U-Net model and DDPM scheduler
3. Train the model with optimized settings
4. Save trained model and generate sample images

Usage:
    python scripts/train.py --help
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
import tyro
import wandb
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader

from config import (
    Config,
    DataConfig,
    ModelConfig,
    SamplingConfig,
    TrainingConfig,
)
from src.data.dataset import create_data_loaders
from src.diffusion.scheduler import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    create_scheduler,
)
from src.models.unet import UNet
from src.training.sampler import DiffusionSampler, create_class_grid_visualization
from src.training.trainer import DiffusionTrainer


def setup_directories(config: Config) -> None:
    """Create necessary output directories."""
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.sample_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    print("Output directories created:")
    print(f"  Models: {config.model_dir}")
    print(f"  Samples: {config.sample_dir}")
    print(f"  Logs: {config.log_dir}")


def create_model_and_scheduler(
    config: Config,
) -> tuple[UNet, DDPMScheduler | DDIMScheduler | PNDMScheduler]:
    """Initialize model and scheduler with configuration."""
    print("Initializing model and scheduler...")

    # Create U-Net model
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

    # Print model info
    total_params = model.count_parameters()
    print(f"Model created with {total_params:,} parameters ({total_params / 1e6:.1f}M)")

    return model, config.scheduler


@dataclass
class TrainingArgs:
    """Training arguments."""

    resume: str = ""
    """Path to checkpoint to resume from."""
    no_sample: bool = False
    """Skip sampling after training."""
    seed: int = 42
    """Random seed."""
    use_wandb: bool = False
    """Use Weights & Biases for logging."""


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def maybe_init_wandb(args: TrainingArgs, config: Config) -> None:
    """Optionally initialize wandb for experiment tracking."""
    if args.use_wandb:
        wandb.init(
            project="mnist-diffusion",
            config=config.__dict__,
            name=f"run-seed-{args.seed}",
            notes="MNIST diffusion training with U-Net and config-driven pipeline",
        )


def print_header(config: Config, args: TrainingArgs) -> None:
    """Print training header."""
    print("=" * 60)
    print("MNIST Diffusion Model Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Model Dir: {config.model_dir}")
    print(f"Sample Dir: {config.sample_dir}")
    print(f"Log Dir: {config.log_dir}")
    print(f"Random Seed: {args.seed}")


def get_data_loaders(
    config: Config,
) -> tuple[DataLoader, DataLoader]:
    """Create data loaders."""
    print("\nLoading MNIST dataset...")
    train_loader, val_loader = create_data_loaders(
        data_root=config.data.data_root,
        img_size=config.data.resize,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    return train_loader, val_loader


def get_trainer(
    model: UNet,
    scheduler: DDPMScheduler | DDIMScheduler | PNDMScheduler,
    config: Config,
) -> DiffusionTrainer:
    """Create trainer."""
    print("\nCreating trainer...")
    return DiffusionTrainer(
        model=model,
        scheduler=scheduler,
        device=config.device,
        **config.training.__dict__,
    )


def maybe_resume_checkpoint(trainer: DiffusionTrainer, args: TrainingArgs) -> None:
    """Optionally resume training from a checkpoint."""
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)


def train_model(
    trainer: DiffusionTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
) -> dict[str, list[float]]:
    """Train model."""
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        num_epochs=config.training.num_epochs,
        val_loader=val_loader,
    )
    print("Training completed!")
    return history


def log_wandb_metrics(history: dict[str, list[float]], args: TrainingArgs) -> None:
    """Log metrics to wandb."""
    if args.use_wandb:
        for epoch, (train_loss, val_loss, lr) in enumerate(
            zip(
                history["train_losses"],
                history["val_losses"]
                if history["val_losses"]
                else [None] * len(history["train_losses"]),  # type: ignore[list-item]
                history["learning_rates"],
                strict=False,
            )
        ):
            metrics = {"train/loss": train_loss, "lr": lr, "epoch": epoch}
            if val_loss is not None:
                metrics["val/loss"] = val_loss
            wandb.log(metrics, step=epoch)


def plot_and_save_curves(
    history: dict[str, list[float]], config: Config, args: TrainingArgs
) -> None:
    """Plot and save training curves."""
    print("\nPlotting training curves...")
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_losses"], label="Train Loss")
    if history["val_losses"]:
        plt.plot(history["val_losses"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(visible=True)

    # Learning rate plot
    plt.subplot(1, 2, 2)
    plt.plot(history["learning_rates"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(visible=True)

    plt.tight_layout()

    # Save training curves
    curves_path = Path(config.log_dir) / "training_curves.png"
    plt.savefig(curves_path)
    print(f"Saved training curves to {curves_path}")
    if args.use_wandb:
        wandb.log({"training_curves": wandb.Image(curves_path)})
    print(f"Training curves saved to {curves_path}")
    plt.show()


def main(args: TrainingArgs, config: Config) -> None:
    """Run training pipeline."""
    set_seed(args.seed)
    maybe_init_wandb(args, config)
    print_header(config, args)
    setup_directories(config)
    train_loader, val_loader = get_data_loaders(config)
    model, scheduler = create_model_and_scheduler(config)
    trainer = get_trainer(model, scheduler, config)
    maybe_resume_checkpoint(trainer, args)
    history = train_model(
        trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    log_wandb_metrics(history, args)
    plot_and_save_curves(history, config, args)

    # Generate samples if not skipped
    if not args.no_sample:
        print("\nGenerating sample images...")

        # Create sampler
        sampler = DiffusionSampler(model, scheduler, config.device)

        # Generate class grid
        sample_path = Path(config.sample_dir) / "generated_samples.png"
        create_class_grid_visualization(
            sampler=sampler,
            samples_per_class=4,
            save_path=str(sample_path),
            img_size=config.model.img_size,
        )

        print(f"Sample images saved to {sample_path}")

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print(f"Model saved in: {config.model_dir}")
    print(f"Samples saved in: {config.sample_dir}")
    print("=" * 60)


def run_train(
    dataset: Literal["mnist"] = "mnist",
    diffusion_mode: Literal[
        "score_matching", "flow_matching", "rectified_flow"
    ] = "score_matching",
    scheduler: Literal["ddpm", "pndm", "ddim"] = "ddpm",
    args: TrainingArgs = TrainingArgs(),  # noqa: B008 # not a fan of this tho
) -> None:
    """Run training pipeline."""
    if dataset == "mnist":
        config = Config(
            scheduler=create_scheduler(scheduler),
            model=ModelConfig(),
            training=TrainingConfig(diffusion_mode=diffusion_mode),
            data=DataConfig(),
            sampling=SamplingConfig(),
        )
        main(args, config)
        config.save_pkl()


if __name__ == "__main__":
    tyro.cli(run_train)
