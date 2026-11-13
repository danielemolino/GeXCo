from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CTNetModel(nn.Module):
    """CT-Net-style classifier for volumetric slice stacks."""

    def __init__(self, n_outputs: int, spatial_pool_size: int = 14):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        self.spatial_pool = nn.AdaptiveAvgPool2d((spatial_pool_size, spatial_pool_size))

        self.reducingconvs = nn.Sequential(
            nn.Conv3d(42, 64, kernel_size=(3, 3, 3), stride=(3, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(3, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=(3, 2, 2), stride=(3, 2, 2), padding=0),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 96),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(96, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.size())
        batch_size = int(shape[0])
        if len(shape) != 5:
            raise ValueError(f"Expected input shape [batch, stacks, 3, H, W], got {shape}")
        num_stacks = shape[1]
        x = x.view(batch_size * num_stacks, shape[2], shape[3], shape[4])
        x = self.features(x)
        x = self.spatial_pool(x)
        x = x.view(batch_size, num_stacks, 512, x.shape[-2], x.shape[-1])
        x = self.reducingconvs(x)
        x = x.view(batch_size, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
        return self.classifier(x)
