"""Training utilities and trainer class."""
import os
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class for model training with MLflow tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        mlflow_tracking: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            mixed_precision: Use mixed precision training
            mlflow_tracking: Enable MLflow tracking
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.mlflow_tracking = mlflow_tracking

        # Mixed precision scaler
        self.scaler = GradScaler("cuda") if mixed_precision and torch.cuda.is_available() else None

        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.mixed_precision and torch.cuda.is_available():
                with autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": running_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                }
            )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate model.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc="Validation")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": running_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                }
            )

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def fit(
        self,
        epochs: int,
        save_dir: str = "models",
        early_stopping_patience: int = 10,
    ) -> Dict[str, list]:
        """
        Train model for specified epochs.

        Args:
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log to MLflow
            if self.mlflow_tracking:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "learning_rate": current_lr,
                    },
                    step=epoch,
                )

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                checkpoint_path = os.path.join(save_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
                print(f"✓ Saved best model with val_acc: {val_acc:.2f}%")

                # Log model checkpoint to MLflow as artifact
                if self.mlflow_tracking:
                    try:
                        mlflow.log_artifact(checkpoint_path, artifact_path="models")
                        print(f"✓ Model checkpoint logged to MLflow")
                    except Exception as e:
                        print(f"Warning: Could not log checkpoint to MLflow: {e}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        return {
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
        }
