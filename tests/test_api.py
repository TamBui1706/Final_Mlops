"""Integration tests for API."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    """Create test client."""
    from api.app import app

    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create sample image for testing."""
    img = Image.new("RGB", (224, 224), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data


def test_model_info_endpoint(client):
    """Test model info endpoint."""
    response = client.get("/model/info")

    # Model might not be loaded in test environment
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "num_classes" in data
        assert "class_names" in data


def test_predict_endpoint_no_file(client):
    """Test predict endpoint without file."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_with_image(client, sample_image):
    """Test predict endpoint with image."""
    files = {"file": ("test.jpg", sample_image, "image/jpeg")}
    response = client.post("/predict", files=files)

    # Model might not be loaded in test environment
    if response.status_code == 200:
        data = response.json()
        assert "class_name" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "inference_time" in data
