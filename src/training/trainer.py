"""Training module for MNIST diffusion model.

This module implements:
- DDPM loss computation with mathematical foundations
- Optimized training loop with mixed precision
- Progress tracking and logging
- Model checkpointing
"""

import random
from dataclasses import field
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from jaxtyping import Float, Int
from pydantic.dataclasses import dataclass
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainingConfig
from src.data.dataset import denormalize_images
from src.diffusion.scheduler import DDIMScheduler, DDPMScheduler, PNDMScheduler
from src.models.unet import UNet
from src.training.sampler import DiffusionSampler


@dataclass(kw_only=True)
class DiffusionTrainer(TrainingConfig):
    """Trainer for DDPM diffusion models.

    Mathematical Background:
    The training objective varies based on the diffusion mode:

    1. Score Matching (DDPM): L = E_{t,x_0,ε} [||ε - ε_θ(x_t, t, y)||²]
    2. Flow Matching: L = E_{t,x_0,x_1} [||v_θ(x_t, t, y) - (x_1 - x_0)||²]
    3. Rectified Flow: L = E_{t,x_0,x_1} [||v_θ(x_t, t, y) - (x_0 - x_t)||²]

    where:
    - t ~ Uniform(0, T): random timestep
    - x_0: clean image from dataset
    - x_1: pure noise ~ N(0, I)
    - x_t: interpolated state
    - ε_θ/v_θ: neural network (U-Net model)
    - y: class label for conditional generation
    """

    model: UNet
    """ U-Net model for noise/velocity prediction. """

    scheduler: DDPMScheduler | DDIMScheduler | PNDMScheduler
    """ DDPM scheduler. """

    optimizer: torch.optim.Optimizer = field(init=False)
    """ Optimizer. """

    device: str = "cuda"
    """ Device for training. """

    scaler: torch.cuda.amp.GradScaler | None = field(init=False)
    """ GradScaler for mixed precision training. """

    train_losses: list[float] = field(init=False)
    """ Training losses. """

    learning_rates: list[float] = field(init=False)
    """ Learning rates. """

    def __post_init__(self) -> None:
        """Initialize the trainer."""
        # Mixed precision scaler
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.mixed_precision and self.device == "cuda"
            else None
        )
        self.optimizer = self.get_optimizer(self.model)
        self._scheduler_lr = self.get_scheduler_lr(self.optimizer)
        self.train_losses = []
        self.learning_rates = []

    def compute_loss(
        self,
        x_0: Float[Tensor, "batch channels height width"],
        y: Int[Tensor, "batch"] | None,  # type: ignore[name-defined]
    ) -> Float[Tensor, ""]:
        """Compute the diffusion model training loss based on selected objective."""
        batch_size = x_0.shape[0]
        diffusion_mode = self.diffusion_mode
        loss_type = self.loss_type

        if diffusion_mode == "score_matching":
            # Standard DDPM: sample timesteps and add noise
            t = torch.randint(
                0,
                self.scheduler.num_timesteps,
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )
            x_t, noise = self.scheduler.q_sample(x_0, t)
            return self._score_matching_loss(x_t, noise, t, y, loss_type)

        if diffusion_mode in ["flow_matching", "rectified_flow"]:
            # Flow-based methods: use continuous time [0, 1]
            t_continuous = torch.rand(batch_size, device=self.device)  # [0, 1]
            x_1 = torch.randn_like(x_0)  # Pure noise

            # Interpolate between x_0 and x_1
            t_expanded = t_continuous.view(-1, 1, 1, 1)
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

            # Convert to discrete timesteps for model input
            t_discrete = (t_continuous * (self.scheduler.num_timesteps - 1)).long()

            if diffusion_mode == "flow_matching":
                return self._flow_matching_loss(x_t, x_0, x_1, t_discrete, y, loss_type)
            # rectified_flow
            return self._rectified_flow_loss(x_t, x_0, x_1, t_discrete, y, loss_type)

        msg = f"Unknown diffusion_mode: {diffusion_mode}"
        raise NotImplementedError(msg)

    def _score_matching_loss(
        self,
        x_t: Float[Tensor, "batch channels height width"],
        noise: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, "batch"],  # type: ignore[name-defined]
        y: Int[Tensor, "batch"],  # type: ignore[name-defined]
        loss_type: str,
    ) -> Float[Tensor, ""]:
        """Score Matching loss (Standard DDPM).

        Mathematical Background:
        L = E_{t,ε} [||ε - ε_θ(x_t, t, y)||^p]
        where p=2 for MSE, p=1 for MAE

        The model predicts the noise ε that was added to create x_t from x_0.
        This is derived from the variational lower bound of the log-likelihood.

        Args:
            x_t: Noisy images at timestep t
            noise: The actual noise that was added
            t: Timestep indices
            y: Class labels
            loss_type: "l1" or "l2" loss

        Returns:
            Loss value

        """
        predicted_noise = self.model(x_t, t, y)
        if loss_type == "l1":
            return F.l1_loss(predicted_noise, noise)
        return F.mse_loss(predicted_noise, noise)

    def _flow_matching_loss(
        self,
        x_t: Float[Tensor, "batch channels height width"],
        x_0: Float[Tensor, "batch channels height width"],
        x_1: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, "batch"],  # type: ignore[name-defined]
        y: Int[Tensor, "batch"],  # type: ignore[name-defined]
        loss_type: str,
    ) -> Float[Tensor, ""]:
        """Flow Matching loss (Lipman et al. 2023).

        Mathematical Background:
        For conditional flow matching with straight paths:
        - Path: ψ_t(x) = (1-t)x_0 + t*x_1
        - Target velocity: u_t(x) = x_1 - x_0
        - Loss: L = E_{t,x_0,x_1} [||v_θ(ψ_t(x), t, y) - u_t(x)||^p]

        The model learns to predict the vector field that transports
        data (x_0) to noise (x_1) along straight paths.

        Args:
            x_t: Interpolated state at time t
            x_0: Clean data
            x_1: Pure noise
            t: Timestep indices (for model input)
            y: Class labels
            loss_type: "l1" or "l2" loss

        Returns:
            Loss value

        """
        v_pred = self.model(x_t, t, y)
        v_target = x_1 - x_0  # Velocity field from data to noise

        if loss_type == "l1":
            return F.l1_loss(v_pred, v_target)
        return F.mse_loss(v_pred, v_target)

    def _rectified_flow_loss(
        self,
        x_t: Float[Tensor, "batch channels height width"],
        x_0: Float[Tensor, "batch channels height width"],
        x_1: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, "batch"],  # type: ignore[name-defined]
        y: Int[Tensor, "batch"] | None,  # type: ignore[name-defined]
        loss_type: str,
    ) -> Float[Tensor, ""]:
        """Rectified Flow loss (Liu et al. 2022).

        Mathematical Background:
        Rectified flow is different from flow matching in its formulation and goal.

        In rectified flow:
        1. We define a REVERSE path from x_1 (noise) to x_0 (data)
        2. The path is: z_t = t*x_0 + (1-t)*x_1, where t ∈ [0,1]
        3. At t=0: z_0 = x_1 (pure noise)
        4. At t=1: z_1 = x_0 (clean data)
        5. The velocity field is: dz_t/dt = x_0 - x_1 (constant)

        However, for training we use the SAME interpolation as flow matching:
        x_t = (1-t)x_0 + t*x_1

        But we predict a DIFFERENT velocity target:
        - We want the model to predict the direction that takes us from the
        current point x_t towards the clean data x_0

        This is NOT simply x_0 - x_1. Instead:
        - At any point x_t on the path, the "rectified" velocity should point
        towards the data endpoint x_0
        - This gives us the target: x_0 - x_t
        - Substituting x_t = (1-t)x_0 + t*x_1:
        v_target = x_0 - x_t = x_0 - [(1-t)x_0 + t*x_1] = t*(x_0 - x_1)

        Loss: L = E_{t,x_0,x_1} [||v_θ(x_t, t, y) - t*(x_0 - x_1)||^2]

        Args:
            x_t: Interpolated state at time t: (1-t)x_0 + t*x_1
            x_0: Clean data sample
            x_1: Pure noise sample
            t: Timestep indices (for model conditioning)
            y: Class labels (can be None for unconditional)
            loss_type: "l1" or "l2" loss

        Returns:
            Loss value

        """
        # Model predicts the velocity field
        v_pred = self.model(x_t, t, y)

        # Convert discrete timesteps back to continuous time for velocity computation
        t_continuous = t.float() / (
            self.scheduler.num_timesteps - 1
        )  # Convert to [0,1]
        t_expanded = t_continuous.view(-1, 1, 1, 1)

        # Rectified flow target: t * (x_0 - x_1)
        # This makes the velocity scale with time - small corrections near x_0, large near x_1
        v_target = t_expanded * (x_0 - x_1)

        if loss_type == "l1":
            return F.l1_loss(v_pred, v_target)
        return F.mse_loss(v_pred, v_target)

    def train_step(
        self,
        batch: tuple[  # type: ignore[name-defined]
            Float[Tensor, "batch channels height width"],
            Int[Tensor, "batch"],
        ],
    ) -> float:
        """Perform a single training step.

        Args:
            batch: Tuple of (images, labels)

        Returns:
            loss: Loss value for this step

        """
        # Extract data
        x_0, y = batch
        x_0 = x_0.to(self.device)
        y_original = y.to(self.device)

        # Classifier-Free Guidance: randomly drop class labels for unconditional training
        if torch.rand(1).item() < self.p_unconditional:
            y_conditional = None
        else:
            y_conditional = y_original

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(x_0, y_conditional)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            loss = self.compute_loss(x_0, y_conditional)
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.optimizer.step()

        # Update learning rate scheduler
        if self._scheduler_lr is not None:
            self._scheduler_lr.step()

        return loss.item()

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            avg_loss: Average loss for the epoch

        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Training step
            loss = self.train_step(batch)
            total_loss += loss

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.6f}",
                }
            )

        avg_loss = total_loss / num_batches

        # Store metrics
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]["lr"])

        return avg_loss

    def train(
        self,
        *,
        train_loader: DataLoader,
        num_epochs: int,
        save_dir: Path = Path("outputs/models"),
        save_every: int = 1,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            save_every: Save checkpoint every N epochs
            val_loader: Optional validation data loader

        Returns:
            training_history: Dictionary with training metrics

        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Diffusion mode: {self.diffusion_mode}")

        # Create save directory
        save_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        history: dict[str, list[float]] = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
        }

        best_loss = float("inf")

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader, epoch)
            history["train_losses"].append(train_loss)

            # Validation (if provided)
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history["val_losses"].append(val_loss)

            # Store learning rate
            history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Visualize denoising steps for a random class/image
            self.visualize_denoising_process(train_loader, epoch)

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                self.save_checkpoint(str(checkpoint_path), epoch, train_loss)

                # Save best model
                current_loss = val_loss if val_loss is not None else train_loss
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_path = save_dir / "best_model.pth"
                    self.save_checkpoint(str(best_path), epoch, current_loss)
                    print(f"  New best model saved (loss: {best_loss:.4f})")

        # Save final model
        final_path = save_dir / "final_model.pth"
        self.save_checkpoint(str(final_path), num_epochs - 1, train_loss)

        print("Training completed!")
        return history

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            avg_val_loss: Average validation loss

        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        for batch in tqdm(val_loader, desc="Validation", leave=False):
            x_0, y = batch
            x_0 = x_0.to(self.device)
            y = y.to(self.device)

            # Compute loss without gradients
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = self.compute_loss(x_0, y)
            else:
                loss = self.compute_loss(x_0, y)

            total_loss += loss.item()

        return total_loss / num_batches

    def save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            loss: Current loss value

        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "scheduler_state_dict": self._scheduler_lr.state_dict()
            if self._scheduler_lr
            else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "train_losses": self.train_losses,
            "learning_rates": self.learning_rates,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(
        self, path: str, *, load_optimizer: bool = True
    ) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state

        Returns:
            checkpoint: Checkpoint dictionary with metadata

        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if (
            self._scheduler_lr
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"]
        ):
            self._scheduler_lr.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state
        if (
            self.scaler
            and "scaler_state_dict" in checkpoint
            and checkpoint["scaler_state_dict"]
        ):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load training history
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        if "learning_rates" in checkpoint:
            self.learning_rates = checkpoint["learning_rates"]

        print(f"Checkpoint loaded from {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Loss: {checkpoint['loss']:.4f}")

        return checkpoint

    def visualize_denoising_process(self, train_loader: DataLoader, epoch: int) -> None:
        """Pick a random class from the dataset, select one image, and show the denoising process.

        from full noise to prediction, saving the visualization and logging to wandb if enabled.
        """
        # Get a batch from the loader
        batch = next(iter(train_loader))
        x_0: Float[Tensor, "batch channels height width"] = batch[0].to(self.device)
        y: Int[Tensor, batch] = batch[1].to(self.device)

        # Pick a random index
        idx = random.randint(0, x_0.shape[0] - 1)
        img = x_0[idx : idx + 1]
        label = y[idx].item()

        # Run the sampling process with intermediates
        sampler = DiffusionSampler(self.model, self.scheduler, self.device)
        _, intermediates = sampler.sample(
            num_samples=1,
            class_labels=torch.tensor([label], device=self.device),
            return_intermediates=True,
            show_progress=False,
            img_size=img.shape[2:],
        )
        # intermediates: list of tensors, length ~num_timesteps
        intermediates = [img.cpu(), *intermediates]  # prepend groundtruth
        intermediates = [denormalize_images(x.cpu()) for x in intermediates]

        # Create GIF of denoising process
        steps = len(intermediates)
        frames = []
        for i in range(steps):
            arr = intermediates[i][0, 0].numpy()  # shape: (H, W)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            frames.append(arr)
        save_dir = Path("outputs/logs")
        save_dir.mkdir(parents=True, exist_ok=True)
        gif_path = save_dir / f"denoise_steps_epoch_{epoch + 1}.gif"
        imageio.mimsave(gif_path, frames, duration=0.5)
        print(f"Denoising process GIF saved to {gif_path}")
        # Log GIF to wandb as video
        if wandb.run is not None:
            wandb.log(
                {
                    f"denoise_steps/epoch_{epoch + 1}": wandb.Video(
                        gif_path, format="gif"
                    )
                }
            )


def create_trainer(
    *,
    model: UNet,
    scheduler: DDPMScheduler,
    device: str = "cuda",
    mixed_precision: bool = True,
    grad_clip_norm: float | None = 1.0,
) -> DiffusionTrainer:
    """Create a configured trainer instance.

    Args:
        model: U-Net model
        scheduler: DDPM scheduler
        device: Device for training
        mixed_precision: Whether to use mixed precision
        grad_clip_norm: Gradient clipping norm

    Returns:
        trainer: Configured trainer instance

    """
    # Create trainer
    return DiffusionTrainer(
        model=model,
        scheduler=scheduler,
        scheduler_lr="cosine",
        device=device,
        mixed_precision=mixed_precision,
        grad_clip_norm=grad_clip_norm,
    )
