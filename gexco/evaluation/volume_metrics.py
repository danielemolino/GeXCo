"""Volumetric PSNR and SSIM used by the GeXCo evaluation suite."""

from __future__ import annotations

import math

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


def _tensor(value: ArrayLike) -> torch.Tensor:
    return value.detach().float() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)


def peak_signal_noise_ratio(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    data_range: float | None = None,
) -> float:
    """Compute PSNR over an entire volume.

    Args:
        prediction: Predicted volume.
        target: Reference volume with the same shape.
        data_range: Intensity range. Inferred from the target when omitted.

    Returns:
        PSNR in decibels, or positive infinity for identical inputs.
    """
    predicted = _tensor(prediction)
    reference = _tensor(target)
    if predicted.shape != reference.shape:
        raise ValueError("prediction and target must have identical shapes.")
    mse = torch.mean((predicted - reference).square()).item()
    if mse == 0.0:
        return float("inf")
    dynamic_range = float(data_range) if data_range is not None else float(reference.max() - reference.min())
    dynamic_range = max(dynamic_range, torch.finfo(torch.float32).eps)
    return 20.0 * math.log10(dynamic_range) - 10.0 * math.log10(mse)


def _batched_spatial_tensor(
    value: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Convert one image or volume to MONAI's batch-channel convention."""
    if value.ndim == 2:
        return value.unsqueeze(0).unsqueeze(0), 2
    if value.ndim == 3:
        return value.unsqueeze(0).unsqueeze(0), 3
    if value.ndim == 4:
        return value.unsqueeze(0), 3
    if value.ndim == 5:
        return value, 3
    raise ValueError("SSIM expects a 2D image or 3D volume with optional batch/channel axes.")


def structural_similarity(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    data_range: float | None = None,
) -> float:
    """Compute MONAI windowed SSIM over an image or volume.

    Args:
        prediction: Predicted volume.
        target: Reference volume with the same shape.
        data_range: Intensity range. Inferred from the target when omitted.

    Returns:
        Scalar SSIM in ``[-1,1]``.
    """
    predicted = _tensor(prediction)
    reference = _tensor(target)
    if predicted.shape != reference.shape:
        raise ValueError("prediction and target must have identical shapes.")
    if torch.equal(predicted, reference):
        return 1.0
    dynamic_range = float(data_range) if data_range is not None else float(reference.max() - reference.min())
    dynamic_range = max(dynamic_range, torch.finfo(torch.float32).eps)
    predicted, spatial_dims = _batched_spatial_tensor(predicted)
    reference, _ = _batched_spatial_tensor(reference)
    from monai.metrics import SSIMMetric

    window_size = min(11, *predicted.shape[-spatial_dims:])
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError("Every spatial axis must contain at least three values for SSIM.")
    metric = SSIMMetric(
        spatial_dims=spatial_dims,
        data_range=dynamic_range,
        win_size=window_size,
    )
    return float(metric(predicted, reference).mean())


def volume_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    data_range: float | None = 2000.0,
) -> dict[str, float]:
    """Compute all core volume metrics.

    Args:
        prediction: Predicted CT.
        target: Reference CT.
        data_range: Optional fixed intensity range.

    Returns:
        Dictionary with ``psnr`` and ``ssim``.
    """
    prediction = _tensor(prediction)
    target = _tensor(target)
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite CT intensities")
    prediction = prediction.clamp(-1000.0, 1000.0)
    target = target.clamp(-1000.0, 1000.0)
    return {
        "psnr": peak_signal_noise_ratio(prediction, target, data_range=data_range),
        "ssim": structural_similarity(prediction, target, data_range=data_range),
    }


# Common concise aliases.
psnr = peak_signal_noise_ratio
ssim = structural_similarity
