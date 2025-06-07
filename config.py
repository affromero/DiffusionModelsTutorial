"""Configuration file for MNIST Diffusion Model.

This file contains all hyperparameters and settings for the diffusion model training.
"""

from typing import Literal

import torch
from pydantic.dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion process.

    Mathematical Background:
    - num_timesteps: T in DDPM paper, controls the length of the diffusion chain
    - beta_schedule: Controls the noise schedule β_t
    - beta_start/end: Range for linear schedule (if used)
    """

    num_timesteps: int = 50  # T: total diffusion steps
    beta_schedule: Literal["linear", "cosine"] = "cosine"  # Cosine gives better results
    beta_start: float = 0.0001  # β_1 for linear schedule
    beta_end: float = 0.02  # β_T for linear schedule
    cosine_s: float = 0.008  # Offset parameter for cosine schedule


@dataclass
class ModelConfig:
    """Configuration for the U-Net model architecture."""

    img_channels: int = 1  # Grayscale images
    img_size: int = 16  # 16x16 resolution (good quality/speed balance)
    time_emb_dim: int = 512  # Time embedding dimension
    num_classes: int = 10  # MNIST digit classes (0-9)


@dataclass
class TrainingConfig:
    """Configuration for training the diffusion model."""

    batch_size: int = 256  # Large batch for stable gradients
    learning_rate: float = 2e-4  # Optimal for AdamW
    weight_decay: float = 0.01  # L2 regularization
    num_epochs: int = 5  # Sufficient for MNIST?

    # Optimization settings
    optimizer: str = "adamw"  # AdamW > Adam for this task
    scheduler: str = "cosine"  # Cosine annealing LR schedule
    grad_clip_norm: float = 1.0  # Gradient clipping for stability

    # Training acceleration
    mixed_precision: bool = True  # 2x speedup on modern GPUs
    compile_model: bool = False  # torch.compile for PyTorch 2.0+

    # Data loading
    num_workers: int = 4  # Parallel data loading
    pin_memory: bool = True  # Faster GPU transfer


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Mathematical Background:
    - normalize: Maps pixel values from [0,1] to [-1,1] for better training
    - resize: 16x16 is optimal balance between quality and computation
    """

    train_val_data_root: str = "data/mnist_train_small.csv"  # Data directory
    test_data_root: str = "data/mnist_test.csv"  # Data directory
    resize: int = 16  # Target image size
    normalize_mean: float = 0.5  # For mapping [0,1] -> [-1,1]
    normalize_std: float = 0.5  # For mapping [0,1] -> [-1,1]


@dataclass
class SamplingConfig:
    """Configuration for sampling from the trained model.

    - num_samples: Number of samples to generate per class
    - eta: Controls stochasticity in DDIM sampling (0=deterministic, 1=DDPM)
    - skip_steps: For faster sampling with DDIM
    """

    num_samples: int = 8  # Samples per digit class
    eta: float = 1.0  # DDPM sampling (fully stochastic)
    skip_steps: int = 1  # Use every timestep (no skipping)
    guidance_scale: float = 1.0  # Classifier-free guidance (1.0 = no guidance)


@dataclass
class Config:
    """Main configuration class combining all settings."""

    diffusion: DiffusionConfig = DiffusionConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    sampling: SamplingConfig = SamplingConfig()

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42  # Reproducibility

    # Output paths
    output_dir: str = "outputs"
    model_dir: str = "outputs/models"
    sample_dir: str = "outputs/samples"
    log_dir: str = "outputs/logs"
