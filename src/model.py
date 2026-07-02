"""CNN architecture for Alzheimer's MRI slice classification."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A small 3-block convolutional network with an adaptive-pool head.

    The adaptive pool makes the classifier independent of the input spatial
    size, so the same model works for any square input resolution.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def build_model(
    num_classes: int = 4,
    dropout: float = 0.5,
    device: torch.device | None = None,
) -> nn.Module:
    """Create a :class:`SimpleCNN` and move it to ``device``.

    Multi-GPU is enabled automatically via :class:`torch.nn.DataParallel` when
    more than one CUDA device is visible.
    """
    model: nn.Module = SimpleCNN(num_classes=num_classes, dropout=dropout)
    if device is not None:
        model = model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model
