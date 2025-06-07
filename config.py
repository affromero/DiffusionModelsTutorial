"""Configuration file for MNIST Diffusion Model.

This file contains all hyperparameters and settings for the diffusion model training.
"""

import pickle
from dataclasses import field
from pathlib import Path
from typing import Literal

import torch
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import transforms

from src.diffusion.scheduler import DDIMScheduler, DDPMScheduler, PNDMScheduler


@dataclass(kw_only=True)
class ModelConfig:
    """Configuration for the U-Net model architecture."""

    img_channels: int = 1  # Grayscale images
    img_size: tuple[int, int] = (
        16,
        16,
    )  # 16x16 resolution (good quality/speed balance)
    time_emb_dim: int = 512  # Time embedding dimension
    num_classes: int = 10  # MNIST digit classes (0-9)

    # U-Net architecture parameters
    base_channels: int = 64  # Base channel count
    channel_mults: tuple = (1, 2, 4)  # Channel multipliers for each level
    num_res_blocks: int = 2  # Residual blocks per level
    attention_resolutions: tuple = (8, 4)  # Resolutions to apply attention
    num_heads: int = 8  # Multi-head attention heads
    dropout: float = 0.1  # Dropout for regularization


@dataclass(kw_only=True)
class TrainingConfig:
    """Configuration for training the diffusion model."""

    batch_size: int = 256  # Large batch for stable gradients
    learning_rate: float = 2e-4  # Optimal for AdamW
    weight_decay: float = 0.01  # L2 regularization
    num_epochs: int = 5  # Sufficient for MNIST?

    # Optimization settings
    optimizer_name: Literal["adamw", "adam"] = "adamw"  # AdamW > Adam for this task
    scheduler_lr: Literal["cosine", "step"] = "cosine"  # Cosine annealing LR schedule
    grad_clip_norm: float | None = 1.0  # Gradient clipping for stability

    # Training acceleration
    mixed_precision: bool = True  # 2x speedup on modern GPUs
    compile_model: bool = False  # torch.compile for PyTorch 2.0+

    # Loss
    loss_type: Literal["l2", "l1"] = "l2"

    # Classifier-Free Guidance training
    p_unconditional: float = 0.1  # Probability of using unconditional training (y=None)

    # Diffusion training objective
    diffusion_mode: Literal["score_matching", "flow_matching", "rectified_flow"] = (
        "score_matching"
    )
    """Training objective: score_matching (DDPM), flow_matching (probability flow ODE), or rectified_flow (RF)."""

    def get_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Get optimizer."""
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        if self.optimizer_name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        msg = f"Unknown optimizer: {self.optimizer_name}"
        raise NotImplementedError(msg)

    def get_scheduler_lr(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Get learning rate scheduler."""
        if self.scheduler_lr == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
            )
        if self.scheduler_lr == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.num_epochs // 2,
                gamma=0.1,
            )
        msg = f"Unknown scheduler: {self.scheduler_lr}"
        raise NotImplementedError(msg)


@dataclass(kw_only=True)
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_root: str = "./data"  # Data directory
    resize: int = 16  # Target image size
    normalize_mean: float = 0.5  # For mapping [0,1] -> [-1,1]
    normalize_std: float = 0.5  # For mapping [0,1] -> [-1,1]
    num_workers: int = 4  # Parallel data loading
    pin_memory: bool = True  # Faster GPU transfer

    train_transform: transforms.Compose = field(
        init=False
    )  # Transformation pipeline for training
    val_transform: transforms.Compose = field(init=False)

    train_dataloader: DataLoader = field(init=False)
    val_dataloader: DataLoader = field(init=False)
    test_dataloader: DataLoader = field(init=False)


@dataclass(kw_only=True)
class SamplingConfig:
    """Configuration for sampling from the trained model.

    - guidance_scale: Controls strength of classifier guidance (1.0 = no guidance)
    """

    guidance_scale: float = 1.0  # Classifier-free guidance (1.0 = no guidance)


@dataclass(kw_only=True)
class Config:
    """Main configuration class combining all settings."""

    scheduler: DDPMScheduler | PNDMScheduler | DDIMScheduler
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    sampling: SamplingConfig

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42  # Reproducibility

    # Output paths
    output_dir: str = "outputs"
    model_dir: str = "outputs/models"
    sample_dir: str = "outputs/samples"
    log_dir: str = "outputs/logs"

    def save_pkl(self) -> None:
        """Save config to pkl file."""
        with (Path(self.output_dir) / "config.pkl").open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pkl(cls, output_dir: str = "outputs") -> "Config":
        """Load config from pkl file."""
        config_pkl = Path(output_dir) / "config.pkl"
        if not config_pkl.exists():
            msg = f"Model not found at {config_pkl}. Please run training first."
            raise FileNotFoundError(msg)
        with config_pkl.open("rb") as f:
            return pickle.load(f)  # noqa: S301
