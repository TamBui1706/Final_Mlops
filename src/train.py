"""Main training script with MLflow tracking."""
import argparse
import os
from datetime import datetime

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from dotenv import load_dotenv

from data import create_dataloaders, load_config
from models import create_model
from training import Trainer
from utils import set_seed

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train rice disease classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--train-dir", type=str, default="train", help="Train directory")
    parser.add_argument("--val-dir", type=str, default="validation", help="Validation directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--model-name", type=str, default=None, help="Model architecture name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    learning_rate = args.lr or config["training"]["learning_rate"]
    model_name = args.model_name or config["model"]["name"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, num_classes, class_names = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=batch_size,
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Classes: {class_names}")

    # Create model
    print(f"\nCreating model: {model_name}")
    model = create_model(
        model_name=model_name,
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"].get("label_smoothing", 0.1))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )

    # MLflow setup
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        # Start MLflow run
        run_name = (
            f"{config['mlflow']['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow.start_run(run_name=run_name)

        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "image_size": config["data"]["image_size"],
                "num_classes": config["model"]["num_classes"],
                "dropout": config["model"]["dropout"],
                "weight_decay": config["training"]["weight_decay"],
                "label_smoothing": config["training"]["label_smoothing"],
                "mixed_precision": config["training"]["mixed_precision"],
            }
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=config["training"]["mixed_precision"],
        mlflow_tracking=use_mlflow,
    )

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        epochs=epochs,
        save_dir=os.getenv("MODEL_SAVE_DIR", "models"),
        early_stopping_patience=config["training"]["early_stopping_patience"],
    )

    # Log final metrics
    if use_mlflow:
        mlflow.log_metric("best_val_acc", trainer.best_val_acc)
        print(f"\n✓ MLflow run completed. Best val accuracy: {trainer.best_val_acc:.2f}%")
        mlflow.end_run()

    print("\n✓ Training completed!")


if __name__ == "__main__":
    main()
