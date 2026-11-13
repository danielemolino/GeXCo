"""CT preprocessing: HU clipping, normalization, and cubic resampling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def clip_and_normalize_hu(
    volume: torch.Tensor,
    hu_range: Sequence[float] = (-1000.0, 1000.0),
) -> torch.Tensor:
    """Clip Hounsfield units and map them to ``[0,1]``.

    Args:
        volume: CT tensor in Hounsfield units.
        hu_range: Inclusive lower and upper clipping values.

    Returns:
        Normalized tensor with the input shape.
    """
    if len(hu_range) != 2 or float(hu_range[0]) >= float(hu_range[1]):
        raise ValueError("hu_range must contain increasing lower and upper bounds.")
    lower, upper = float(hu_range[0]), float(hu_range[1])
    clipped = volume.clamp(lower, upper)
    return (clipped - lower) / (upper - lower)


def denormalize_hu(
    volume: torch.Tensor,
    hu_range: Sequence[float] = (-1000.0, 1000.0),
) -> torch.Tensor:
    """Map a normalized MAISI CT back to clipped Hounsfield units.

    Args:
        volume: Normalized CT tensor, nominally in ``[0,1]``.
        hu_range: Output HU interval.

    Returns:
        CT tensor in Hounsfield units.
    """
    if len(hu_range) != 2 or float(hu_range[0]) >= float(hu_range[1]):
        raise ValueError("hu_range must contain increasing lower and upper bounds.")
    lower, upper = float(hu_range[0]), float(hu_range[1])
    return volume.clamp(0.0, 1.0) * (upper - lower) + lower


def resample_volume(volume: torch.Tensor, output_size: int | Sequence[int] = 128) -> torch.Tensor:
    """Trilinearly resample a CT tensor.

    Args:
        volume: Tensor shaped ``(B,C,D,H,W)`` or ``(C,D,H,W)``.
        output_size: Integer cube edge or ``(D,H,W)`` shape.

    Returns:
        Resampled tensor preserving whether a batch dimension was supplied.
    """
    squeeze_batch = volume.ndim == 4
    if squeeze_batch:
        volume = volume.unsqueeze(0)
    if volume.ndim != 5:
        raise ValueError("volume must have shape (C,D,H,W) or (B,C,D,H,W).")
    size = (output_size,) * 3 if isinstance(output_size, int) else tuple(int(v) for v in output_size)
    if len(size) != 3:
        raise ValueError("output_size must contain three spatial dimensions.")
    result = F.interpolate(volume.float(), size=size, mode="trilinear", align_corners=False)
    return result.squeeze(0) if squeeze_batch else result


def preprocess_volume(
    volume: torch.Tensor,
    *,
    output_size: int | Sequence[int] = 128,
    hu_range: Sequence[float] = (-1000.0, 1000.0),
) -> torch.Tensor:
    """Apply the paper's CT resampling and HU normalization contract.

    Args:
        volume: CT tensor in Hounsfield units.
        output_size: Target spatial shape.
        hu_range: HU clipping interval.

    Returns:
        Resampled, normalized CT tensor.
    """
    return clip_and_normalize_hu(resample_volume(volume, output_size), hu_range)
