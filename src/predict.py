"""Prediction script for single image or batch inference."""
import argparse
import os
from typing import List

import cv2
import torch
from data.transforms import get_val_transforms
from models import create_model
from utils import set_seed


def load_model(model_path: str, model_name: str = "efficientnet_b0", num_classes: int = 6):
    """Load trained model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        model_name=model_name, num_classes=num_classes, pretrained=False
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    transforms,
    class_names: List[str],
):
    """Predict single image."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    augmented = transforms(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score, probabilities


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Predict rice disease")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-name", type=str, default="efficientnet_b0", help="Model architecture"
    )
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Class names
    class_names = [
        "bacterial_leaf_blight",
        "brown_spot",
        "healthy",
        "leaf_blast",
        "leaf_scald",
        "narrow_brown_spot",
    ]

    # Load model
    print(f"Loading model from {args.model}")
    model, device = load_model(args.model, args.model_name, len(class_names))
    print(f"Using device: {device}")

    # Load transforms
    transforms = get_val_transforms(args.image_size)

    # Predict
    print(f"\nPredicting: {args.image}")
    predicted_class, confidence, probabilities = predict_image(
        args.image, model, device, transforms, class_names
    )

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"{'=' * 50}")
    print("\nAll probabilities:")
    for class_name, prob in zip(class_names, probabilities):
        print(f"  {class_name:25s}: {prob.item():.2%}")


if __name__ == "__main__":
    main()
