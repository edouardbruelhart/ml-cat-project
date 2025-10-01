from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models  # type: ignore[import-untyped]


# Test codecov integration
class SimpleNN(nn.Module):
    """
    Simple neural network using a pretrained ResNet18 backbone.
    Designed for 2-class classification: cat vs not-cat.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Load ResNet18 backbone
        self.backbone = models.resnet18()
        # Replace the final fully connected layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Any:
        """
        Simple forward pass through the network.
        """
        return self.backbone(x)
