"""Geometry-aware lifting of 2D X-ray features into a 3D latent grid."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gexco.reconstruction.geometry import (
    ViewMetadata,
    normalize_detector_pixels,
    project_points_to_detector,
)


def lift_view_features(
    feature_map: torch.Tensor,
    latent_grid: torch.Tensor,
    view: ViewMetadata,
) -> torch.Tensor:
    """Sample a 2D feature map along the rays of a 3D voxel grid.

    Args:
        feature_map: Encoded view shaped ``(B, C, Hf, Wf)``.
        latent_grid: World coordinates shaped ``(B, D, H, W, 3)``.
        view: Physical projection geometry.

    Returns:
        Lifted features shaped ``(B, C, D, H, W)``.
    """
    if feature_map.ndim != 4 or latent_grid.ndim != 5 or latent_grid.shape[-1] != 3:
        raise ValueError("Expected feature_map (B,C,H,W) and latent_grid (B,D,H,W,3).")
    if feature_map.shape[0] != latent_grid.shape[0]:
        raise ValueError("feature_map and latent_grid batch sizes must match.")
    batch_size, channels = feature_map.shape[:2]
    depth, height, width = latent_grid.shape[1:4]
    pixels = project_points_to_detector(latent_grid, view)
    normalized = normalize_detector_pixels(pixels, view)
    finite = torch.isfinite(normalized).all(dim=-1)
    valid = finite & (normalized.abs() <= 1.0).all(dim=-1)
    normalized = torch.nan_to_num(normalized, nan=2.0, posinf=2.0, neginf=-2.0)

    # Treat depth as an output batch dimension and sample the same 2D map for
    # each slab. This is equivalent to ray-wise feature replication.
    maps = feature_map[:, None].expand(-1, depth, -1, -1, -1)
    maps = maps.reshape(batch_size * depth, channels, *feature_map.shape[-2:])
    grids = normalized.reshape(batch_size * depth, height, width, 2)
    sampled = F.grid_sample(
        maps,
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    lifted = sampled.reshape(batch_size, depth, channels, height, width)
    lifted = lifted.permute(0, 2, 1, 3, 4).contiguous()
    return lifted * valid.unsqueeze(1).to(dtype=feature_map.dtype)
