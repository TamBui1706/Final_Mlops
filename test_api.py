"""Test API prediction with sample images."""
import glob
import json
import os
import random
from pathlib import Path

import requests

# API endpoint
API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 80)
    print("🏥 HEALTH CHECK")
    print("=" * 80)

    response = requests.get(f"{API_URL}/health")
    data = response.json()

    print(f"Status: {data['status']}")
    print(f"Model Loaded: {data['model_loaded']}")
    print(f"Device: {data['device']}")

    return data["status"] == "healthy"


def test_prediction(image_path: str):
    """Test prediction on single image."""
    print("\n" + "=" * 80)
    print(f"🔮 PREDICTION TEST")
    print("=" * 80)
    print(f"Image: {image_path}")

    # Get true label from path
    true_label = Path(image_path).parent.name
    print(f"True Label: {true_label}")

    # Make prediction
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/predict", files=files)

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Prediction: {data['class_name']}")
        print(f"  Confidence: {data['confidence']:.2%}")
        print(f"  Inference Time: {data['inference_time']:.4f}s")
        print(f"  Correct: {'✓' if data['class_name'] == true_label else '✗'}")

        print("\nTop 3 Probabilities:")
        sorted_probs = sorted(data["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
        for cls, prob in sorted_probs:
            print(f"  {cls}: {prob:.2%}")

        return data["class_name"] == true_label
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return False


def test_model_info():
    """Test model info endpoint."""
    print("\n" + "=" * 80)
    print("ℹ️ MODEL INFO")
    print("=" * 80)

    response = requests.get(f"{API_URL}/model/info")
    if response.status_code == 200:
        data = response.json()
        print(f"Model: {data.get('model_name', 'N/A')}")
        print(f"Classes: {data.get('num_classes', 0)}")
        print(f"Class Names: {', '.join(data.get('class_names', []))}")
        print(f"Total Parameters: {data.get('total_parameters', 0):,}")
        print(f"Trainable Parameters: {data.get('trainable_parameters', 0):,}")
        print(f"Device: {data.get('device', 'N/A')}")
    else:
        print(f"✗ Error: {response.status_code}")


def test_multiple_images(num_samples: int = 5):
    """Test prediction on multiple random images."""
    print("\n" + "=" * 80)
    print(f"🎯 TESTING {num_samples} RANDOM IMAGES")
    print("=" * 80)

    # Get random images from validation set
    val_images = []
    for class_dir in Path("validation").iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            val_images.extend(images)

    if not val_images:
        print("✗ No validation images found!")
        return

    # Sample random images
    sample_images = random.sample(val_images, min(num_samples, len(val_images)))

    correct = 0
    total_time = 0

    for i, img_path in enumerate(sample_images, 1):
        print(f"\n--- Image {i}/{num_samples} ---")
        is_correct = test_prediction(str(img_path))
        if is_correct:
            correct += 1

    print("\n" + "=" * 80)
    print(f"📊 SUMMARY")
    print("=" * 80)
    print(f"Accuracy: {correct}/{num_samples} ({correct/num_samples*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    # Test health
    if not test_health():
        print("\n✗ API is not healthy!")
        exit(1)

    # Test model info
    test_model_info()

    # Test single image (get first available)
    val_images = list(Path("validation").rglob("*.jpg"))
    if val_images:
        test_prediction(str(val_images[0]))

    # Test multiple images
    test_multiple_images(num_samples=5)
