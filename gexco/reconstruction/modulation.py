"""Adaptive normalization and volumetric conditioning modules."""

from __future__ import annotations

import torch
from torch import nn


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm whose conditioning branch is initialized to zero."""

    def __init__(self, hidden_dim: int, condition_dim: int) -> None:
        """Initialize AdaLN-Zero.

        Args:
            hidden_dim: Feature dimension to normalize.
            condition_dim: Conditioning-vector dimension.
        """
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale_shift_gate = nn.Linear(condition_dim, 3 * hidden_dim)
        nn.init.zeros_(self.to_scale_shift_gate.weight)
        nn.init.zeros_(self.to_scale_shift_gate.bias)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Modulate normalized features.

        Args:
            hidden: Tensor shaped ``(B,N,C)``.
            condition: Tensor shaped ``(B,condition_dim)``.

        Returns:
            Modulated features and a residual gate shaped ``(B,1,C)``.
        """
        scale, shift, gate = self.to_scale_shift_gate(condition).chunk(3, dim=-1)
        normalized = self.normalization(hidden)
        modulated = normalized * (1.0 + scale[:, None]) + shift[:, None]
        return modulated, gate[:, None]


class ConditioningInjection3D(nn.Module):
    """Inject spatial GeXCo conditioning into a denoiser feature map."""

    def __init__(self, conditioning_channels: int, hidden_channels: int) -> None:
        """Initialize zero-convolution conditioning injection.

        Args:
            conditioning_channels: Input conditioning channels.
            hidden_channels: Denoiser feature channels.
        """
        super().__init__()
        self.projection = nn.Conv3d(conditioning_channels, hidden_channels, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Add projected conditioning to a hidden volume.

        Args:
            hidden: Denoiser feature volume.
            conditioning: Aligned GeXCo conditioning volume.

        Returns:
            Conditioned hidden volume.
        """
        return hidden + self.projection(conditioning)


class FiLMModulation3D(nn.Module):
    """Pooled volumetric feature-wise affine modulation."""

    def __init__(self, conditioning_channels: int, target_channels: int) -> None:
        """Initialize volumetric scale and bias predictors.

        Args:
            conditioning_channels: Channels in the fused condition.
            target_channels: Channels in the modulated latent.
        """
        super().__init__()
        self.to_gamma = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(conditioning_channels, target_channels, kernel_size=1),
        )
        self.to_beta = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(conditioning_channels, target_channels, kernel_size=1),
        )

    def forward(self, hidden: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Apply feature-wise affine modulation.

        Args:
            hidden: Target tensor shaped ``(B,C,D,H,W)``.
            conditioning: Conditioning tensor shaped ``(B,Cc,D,H,W)``.

        Returns:
            Modulated tensor with the same shape as ``hidden``.
        """
        gamma = self.to_gamma(conditioning)
        beta = self.to_beta(conditioning)
        return hidden * (1.0 + gamma) + beta


__all__ = ["AdaLNZero", "ConditioningInjection3D", "FiLMModulation3D"]
