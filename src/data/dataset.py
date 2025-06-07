"""Data loading and preprocessing for MNIST diffusion model training."""

from dataclasses import field
from typing import Any

import torch
import torchvision
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import transforms

from config import DataConfig


@dataclass
class MNISTDataModule(DataConfig):
    """Data module for MNIST dataset with diffusion-optimized preprocessing.

    Inherits all configuration fields from DataConfig.
    """

    train_set: torchvision.datasets.MNIST = field(init=False)
    val_set: torchvision.datasets.MNIST = field(init=False)
    test_set: torchvision.datasets.MNIST = field(init=False)

    def __post_init__(self) -> None:
        """Initialize transforms and datasets."""
        self.train_transform = self._create_transforms()
        self.val_transform = self._create_transforms()  # Same as train for diffusion

    def _create_transforms(self) -> transforms.Compose:
        """Create transformation pipeline for MNIST images using self config values."""
        transform_list = [
            transforms.Resize(self.resize),
            transforms.ToTensor(),
        ]
        if hasattr(self, "normalize_mean") and hasattr(self, "normalize_std"):
            transform_list.append(
                transforms.Normalize(
                    mean=[self.normalize_mean], std=[self.normalize_std]
                )
            )
        return transforms.Compose(transform_list)

    def setup(self) -> None:
        """Set up train and validation datasets."""
        self.train_set = torchvision.datasets.MNIST(
            root=self.data_root,
            train=True,
            transform=self.train_transform,
            download=True,
        )

        self.val_set = torchvision.datasets.MNIST(
            root=self.data_root,
            train=False,
            transform=self.val_transform,
            download=True,
        )

    def get_train_dataloader(self, batch_size: int) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # Ensure consistent batch sizes
        )

    def get_val_dataloader(self, batch_size: int) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def get_dataset_info(self) -> dict:
        """Get information about the dataset."""
        if not hasattr(self, "train_set"):
            self.setup()

        return {
            "num_classes": 10,  # MNIST has 10 digit classes
            "img_channels": 1,  # Grayscale images
            "img_size": self.resize,
            "train_size": len(self.train_set),
            "val_size": len(self.val_set),
            "class_names": [str(i) for i in range(10)],  # '0', '1', ..., '9'
        }


def create_data_loaders(
    *, batch_size: int, **kwargs: Any
) -> tuple[DataLoader, DataLoader]:
    """Get MNIST data loaders using DataConfig fields."""
    data_module = MNISTDataModule(**kwargs)
    data_module.setup()
    return data_module.get_train_dataloader(
        batch_size=batch_size
    ), data_module.get_val_dataloader(batch_size=batch_size)


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Denormalize images from [-1, 1] back to [0, 1] for visualization."""
    return images * 0.5 + 0.5


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Normalize images from [0, 1] to [-1, 1] range."""
    return images * 2.0 - 1.0


if __name__ == "__main__":
    train_loader, val_loader = create_data_loaders(batch_size=32)
    dataset_info = MNISTDataModule().get_dataset_info()
    print(dataset_info)
