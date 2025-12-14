"""Data loading utilities."""
from typing import List, Tuple

import yaml
from torch.utils.data import DataLoader

from .dataset import RiceDataset
from .transforms import get_train_transforms, get_val_transforms


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """
    Create train and validation dataloaders.

    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, num_classes, class_names)
    """
    # Get transforms
    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    # Create datasets
    train_dataset = RiceDataset(train_dir, transform=train_transforms)
    val_dataset = RiceDataset(
        val_dir, transform=val_transforms, class_names=train_dataset.class_names
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    num_classes = len(train_dataset.class_names)
    class_names = train_dataset.class_names

    return train_loader, val_loader, num_classes, class_names
