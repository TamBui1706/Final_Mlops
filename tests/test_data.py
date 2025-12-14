"""Unit tests for data module."""
import os
import tempfile

import pytest
import torch

from src.data import RiceDataset, create_dataloaders, get_train_transforms, get_val_transforms


@pytest.fixture
def sample_data_dir():
    """Create temporary sample data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create class directories
        classes = ["healthy", "diseased"]
        for cls in classes:
            class_dir = os.path.join(tmpdir, cls)
            os.makedirs(class_dir, exist_ok=True)

            # Create dummy images
            import numpy as np
            from PIL import Image

            for i in range(5):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(class_dir, f"img_{i}.jpg"))

        yield tmpdir


def test_rice_dataset(sample_data_dir):
    """Test RiceDataset creation."""
    dataset = RiceDataset(sample_data_dir, transform=None)

    assert len(dataset) == 10
    assert len(dataset.class_names) == 2
    assert "healthy" in dataset.class_names
    assert "diseased" in dataset.class_names


def test_dataset_getitem(sample_data_dir):
    """Test dataset __getitem__ method."""
    transforms = get_val_transforms(224)
    dataset = RiceDataset(sample_data_dir, transform=transforms)

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < 2


def test_train_transforms():
    """Test training transforms."""
    transforms = get_train_transforms(224)

    assert transforms is not None


def test_val_transforms():
    """Test validation transforms."""
    transforms = get_val_transforms(224)

    assert transforms is not None


def test_class_distribution(sample_data_dir):
    """Test class distribution calculation."""
    dataset = RiceDataset(sample_data_dir)
    distribution = dataset.get_class_distribution()

    assert isinstance(distribution, dict)
    assert len(distribution) == 2
    assert distribution["healthy"] == 5
    assert distribution["diseased"] == 5
