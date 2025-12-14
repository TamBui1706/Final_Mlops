"""Unit tests for model module."""
import pytest
import torch

from src.models import RiceClassifier, create_model


def test_create_model():
    """Test model creation."""
    model = create_model(
        model_name="efficientnet_b0",
        num_classes=6,
        pretrained=False,
        dropout=0.3,
    )

    assert isinstance(model, RiceClassifier)
    assert model.num_classes == 6
    assert model.model_name == "efficientnet_b0"


def test_model_forward():
    """Test model forward pass."""
    model = create_model(
        model_name="efficientnet_b0",
        num_classes=6,
        pretrained=False,
    )

    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        output = model(x)

    assert output.shape == (batch_size, 6)


def test_model_parameters():
    """Test model parameter counting."""
    model = create_model(
        model_name="efficientnet_b0",
        num_classes=6,
        pretrained=False,
    )

    total_params = model.get_num_params()
    trainable_params = model.get_trainable_params()

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params


@pytest.mark.parametrize("num_classes", [2, 6, 10])
def test_different_num_classes(num_classes):
    """Test model with different number of classes."""
    model = create_model(
        model_name="efficientnet_b0",
        num_classes=num_classes,
        pretrained=False,
    )

    assert model.num_classes == num_classes

    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, num_classes)
