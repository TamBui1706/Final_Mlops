"""Data processing utilities for rice disease classification."""
import os
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class RiceDataset(Dataset):
    """Custom dataset for rice leaf disease images."""

    def __init__(
        self,
        root_dir: str,
        transform: A.Compose = None,
        class_names: List[str] = None,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory containing class folders
            transform: Albumentations transforms
            class_names: List of class names (folder names)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = class_names or self._get_class_names()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.samples = self._make_dataset()

    def _get_class_names(self) -> List[str]:
        """Get class names from directory structure."""
        classes = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        return sorted(classes)

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """Create list of (image_path, class_idx) tuples."""
        samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                if os.path.splitext(filename)[1].lower() in valid_extensions:
                    path = os.path.join(class_dir, filename)
                    samples.append((path, class_idx))

        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class."""
        distribution = {cls: 0 for cls in self.class_names}
        for _, label in self.samples:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        return distribution
