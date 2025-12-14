"""Model evaluation script."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from data import create_dataloaders, load_config
from models import create_model
from utils import set_seed


def evaluate_model(model, val_loader, device, class_names, save_dir="evaluation_results"):
    """Evaluate model and generate metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'=' * 50}\n")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("Classification Report:")
    print(report)

    # Save results
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics to JSON
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        ),
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    print(f"✓ Saved confusion matrix to {save_dir}/confusion_matrix.png")

    # Per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_accuracies)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_accuracy.png"), dpi=300)
    print(f"✓ Saved per-class accuracy to {save_dir}/per_class_accuracy.png")

    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate rice disease classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--val-dir", type=str, default="validation", help="Validation directory")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloader
    print("Loading data...")
    _, val_loader, num_classes, class_names = create_dataloaders(
        train_dir="train",  # Not used but required
        val_dir=args.val_dir,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
    )

    print(f"Classes: {class_names}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = create_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=False,
        dropout=config["model"]["dropout"],
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print("✓ Model loaded")

    # Evaluate
    metrics = evaluate_model(model, val_loader, device, class_names, args.save_dir)

    print(f"\n✓ Evaluation completed! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
