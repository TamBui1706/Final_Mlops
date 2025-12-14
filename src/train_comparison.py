"""Train and compare multiple model configurations."""
import argparse
import os
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Train and compare rice disease classifiers")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--train-dir", type=str, default="train", help="Train directory")
    parser.add_argument("--val-dir", type=str, default="validation", help="Validation directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip configs with existing models"
    )
    parser.add_argument(
        "--start-from", type=int, default=0, help="Start from config index (0, 1, or 2)"
    )
    return parser.parse_args()


# Define 3 configurations to compare
CONFIGS = [
    {
        "name": "efficientnet_b0_baseline",
        "model_name": "efficientnet_b0",
        "learning_rate": 0.001,
        "batch_size": 32,
        "weight_decay": 0.0001,
        "description": "EfficientNet-B0 with baseline hyperparameters",
    },
    {
        "name": "efficientnet_b0_optimized",
        "model_name": "efficientnet_b0",
        "learning_rate": 0.0005,
        "batch_size": 16,
        "weight_decay": 0.00005,
        "description": "EfficientNet-B0 with lower LR and smaller batch",
    },
    {
        "name": "mobilenetv3_large",
        "model_name": "mobilenetv3_large_100",
        "learning_rate": 0.001,
        "batch_size": 32,
        "weight_decay": 0.0001,
        "description": "MobileNetV3-Large for faster inference",
    },
]


def train_single_config(
    config_dict, base_config, train_dir, val_dir, epochs, device, seed, run_index
):
    """
    Train a single configuration.

    Args:
        config_dict: Configuration dictionary
        base_config: Base configuration from yaml
        train_dir: Training data directory
        val_dir: Validation data directory
        epochs: Number of training epochs
        device: Device to train on
        seed: Random seed
        run_index: Index of this run

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Configuration {run_index + 1}/3: {config_dict['name']}")
    print(f"Description: {config_dict['description']}")
    print(f"{'='*80}\n")

    # Set seed
    set_seed(seed)

    # Create dataloaders
    print(f"Loading data with batch_size={config_dict['batch_size']}...")
    train_loader, val_loader, num_classes, class_names = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=config_dict["batch_size"],
        num_workers=base_config.get("data", {}).get("num_workers", 4),
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Classes: {class_names}\n")

    # Create model
    print(f"Creating model: {config_dict['model_name']}")
    model = create_model(
        model_name=config_dict["model_name"], num_classes=num_classes, pretrained=True
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_dict["learning_rate"],
        weight_decay=config_dict["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config_dict["learning_rate"] * 0.01
    )

    # MLflow setup
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", base_config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(base_config["mlflow"]["experiment_name"])

    # Start MLflow run
    run_name = f"{config_dict['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log all parameters
        mlflow.log_params(
            {
                "model_name": config_dict["model_name"],
                "learning_rate": config_dict["learning_rate"],
                "batch_size": config_dict["batch_size"],
                "weight_decay": config_dict["weight_decay"],
                "epochs": epochs,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "num_classes": num_classes,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "config_name": config_dict["name"],
                "description": config_dict["description"],
                "seed": seed,
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
            mixed_precision=False,  # Disable for CPU
            mlflow_tracking=True,
        )

        # Train
        print("Starting training...\n")
        save_dir = Path("models") / config_dict["name"]
        save_dir.mkdir(parents=True, exist_ok=True)

        history = trainer.fit(epochs=epochs, save_dir=str(save_dir), early_stopping_patience=10)

        # Get best metrics
        best_val_acc = max(history["val_accs"])
        best_val_loss = min(history["val_losses"])
        final_train_acc = history["train_accs"][-1]
        final_train_loss = history["train_losses"][-1]

        # Log final metrics
        mlflow.log_metrics(
            {
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "final_train_acc": final_train_acc,
                "final_train_loss": final_train_loss,
            }
        )

        # Log final model checkpoint
        checkpoint_path = save_dir / "best_model.pth"
        try:
            mlflow.log_artifact(str(checkpoint_path), artifact_path="final_model")
            print("✓ Final model logged to MLflow")
        except Exception as e:
            print(f"Warning: Could not log final model to MLflow: {e}")

        print(f"\n{'='*80}")
        print(f"Configuration {config_dict['name']} completed!")
        print(f"Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"{'='*80}\n")

        run_id = mlflow.active_run().info.run_id

        return {
            "config_name": config_dict["name"],
            "model_name": config_dict["model_name"],
            "learning_rate": config_dict["learning_rate"],
            "batch_size": config_dict["batch_size"],
            "weight_decay": config_dict["weight_decay"],
            "description": config_dict["description"],
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "final_train_acc": final_train_acc,
            "final_train_loss": final_train_loss,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "run_id": run_id,
            "checkpoint_path": str(checkpoint_path),
        }


def main():
    """Main training comparison function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Train all configurations
    results = []
    for i, config_dict in enumerate(CONFIGS):
        # Skip if requested to start from later config
        if i < args.start_from:
            print(f"Skipping Configuration {i+1}/3: {config_dict['name']}")
            continue

        # Check if model already exists and skip if requested
        model_path = Path("models") / config_dict["name"] / "best_model.pth"
        if args.skip_existing and model_path.exists():
            print(f"\n{'='*80}")
            print(f"Configuration {i+1}/3: {config_dict['name']}")
            print(f"Model already exists at {model_path}. Skipping...")
            print(f"{'='*80}\n")
            continue

        result = train_single_config(
            config_dict=config_dict,
            base_config=config,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            epochs=args.epochs,
            device=device,
            seed=args.seed,
            run_index=i,
        )
        results.append(result)

    # Check if any configs were trained
    if not results:
        print("\n" + "=" * 120)
        print("⚠️  All configurations already exist. No new training performed.")
        print("=" * 120)
        print("\nTo retrain:")
        print("  - Remove --skip-existing flag")
        print("  - Delete model folders: models/<config_name>/")
        print("  - Or view existing results in evaluation_results/ folder")
        print("=" * 120)
        return

    # Create comparison dataframe
    df_results = pd.DataFrame(results)

    # Sort by best validation accuracy
    df_results = df_results.sort_values("best_val_acc", ascending=False)

    # Save results
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"model_comparison_{timestamp}.csv"
    df_results.to_csv(results_path, index=False)

    # Print comparison table
    print("\n" + "=" * 120)
    print("MODEL COMPARISON RESULTS")
    print("=" * 120)
    print(f"\n{df_results.to_string(index=False)}\n")

    # Print winner
    best_config = df_results.iloc[0]
    print("=" * 120)
    print("🏆 BEST CONFIGURATION")
    print("=" * 120)
    print(f"Config Name: {best_config['config_name']}")
    print(f"Model: {best_config['model_name']}")
    print(f"Description: {best_config['description']}")
    print(f"Best Val Accuracy: {best_config['best_val_acc']:.2f}%")
    print(f"Best Val Loss: {best_config['best_val_loss']:.4f}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Total Parameters: {best_config['total_params']:,}")
    print(f"Model saved at: {best_config['checkpoint_path']}")
    print(f"MLflow Run ID: {best_config['run_id']}")
    print("=" * 120)

    # Print summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 120)
    print(f"Average Validation Accuracy: {df_results['best_val_acc'].mean():.2f}%")
    print(f"Std Validation Accuracy: {df_results['best_val_acc'].std():.2f}%")
    print(f"Best Accuracy: {df_results['best_val_acc'].max():.2f}%")
    print(f"Worst Accuracy: {df_results['best_val_acc'].min():.2f}%")
    print(
        f"Accuracy Range: {df_results['best_val_acc'].max() - df_results['best_val_acc'].min():.2f}%"
    )
    print("-" * 120)

    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Models saved in: models/<config_name>/best_model.pt")
    print(f"\nView MLflow UI at: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")


if __name__ == "__main__":
    main()
