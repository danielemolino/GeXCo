"""Per-voxel fusion modules used by GeXCo geometry-aware conditioning."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _group_count(channels: int) -> int:
    """Choose the largest small GroupNorm divisor."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class AverageViewFusion(nn.Module):
    """Element-wise average over lifted view volumes."""

    def forward(self, lifted_views: Sequence[torch.Tensor]) -> torch.Tensor:
        """Fuse lifted views uniformly.

        Args:
            lifted_views: Non-empty sequence of ``(B,C,D,H,W)`` tensors.

        Returns:
            Element-wise mean with shape ``(B,C,D,H,W)``.
        """
        if not lifted_views:
            raise ValueError("At least one lifted view is required.")
        return torch.stack(tuple(lifted_views), dim=1).mean(dim=1)


class WeightedViewFusion3D(nn.Module):
    """Checkpoint-compatible learned fusion with voxel-wise softmax weights."""

    def __init__(self, in_channels_per_view: int, out_channels: int) -> None:
        """Initialize the shared projection and scalar weight head.

        Args:
            in_channels_per_view: Channels in every lifted view.
            out_channels: Channels in the fused volume.
        """
        super().__init__()
        self.view_proj = nn.Sequential(
            nn.Conv3d(
                in_channels_per_view,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_group_count(out_channels),
                num_channels=out_channels,
            ),
            nn.SiLU(),
        )
        self.weight_head = nn.Conv3d(out_channels, 1, kernel_size=1)

    def forward(
        self,
        lifted_views: Sequence[torch.Tensor],
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Project and adaptively fuse lifted views.

        Args:
            lifted_views: Non-empty sequence of ``(B,C,D,H,W)`` tensors.
            return_weights: Also return normalized per-view voxel weights.

        Returns:
            Fused volume, optionally paired with weights shaped
            ``(B,V,1,D,H,W)``.
        """
        if not lifted_views:
            raise ValueError("At least one lifted view is required.")
        projected = [self.view_proj(view) for view in lifted_views]
        logits = torch.stack([self.weight_head(view) for view in projected], dim=1)
        weights = torch.softmax(logits, dim=1)
        projected_tensor = torch.stack(projected, dim=1)
        fused = (weights * projected_tensor).sum(dim=1)
        return (fused, weights) if return_weights else fused


class AdaptiveViewFusion(WeightedViewFusion3D):
    """User-facing alias accepting an optional output channel count."""

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        """Initialize adaptive fusion.

        Args:
            in_channels: Per-view channels.
            out_channels: Fused channels, defaulting to ``in_channels``.
        """
        super().__init__(in_channels, int(out_channels or in_channels))


AveragePoolFusion3D = AverageViewFusion

__all__ = [
    "AdaptiveViewFusion",
    "AveragePoolFusion3D",
    "AverageViewFusion",
    "WeightedViewFusion3D",
]
