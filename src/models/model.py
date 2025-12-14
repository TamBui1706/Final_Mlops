"""Model architecture for rice disease classification."""
import timm
import torch
import torch.nn as nn


class RiceClassifier(nn.Module):
    """Rice disease classifier based on pretrained models."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize model.

        Args:
            model_name: Name of the timm model
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            dropout: Dropout rate
        """
        super(RiceClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained model
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get feature dimension by running a dummy forward pass
        # This is the most reliable way across different architectures
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            if len(dummy_output.shape) == 4:  # [B, C, H, W]
                self.feature_dim = dummy_output.shape[1]
            elif len(dummy_output.shape) == 2:  # [B, features] (already pooled)
                self.feature_dim = dummy_output.shape[1]
            else:
                raise ValueError(f"Unexpected output shape: {dummy_output.shape}")

        print(f"Backbone feature dimension: {self.feature_dim}")

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)

        # Global pooling
        features = self.global_pool(features)
        features = features.flatten(1)

        # Classification
        logits = self.classifier(features)

        return logits

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 6,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> RiceClassifier:
    """
    Create model instance.

    Args:
        model_name: Name of the timm model
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate

    Returns:
        Model instance
    """
    model = RiceClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
    return model
