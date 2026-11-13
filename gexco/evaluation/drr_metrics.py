"""Reprojection PSNR/SSIM for frontal and lateral DRRs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from gexco.evaluation.volume_metrics import peak_signal_noise_ratio, structural_similarity


ArrayLike = np.ndarray | torch.Tensor


def reprojection_metrics(
    predicted_drrs: Sequence[ArrayLike],
    target_drrs: Sequence[ArrayLike],
    *,
    data_range: float | None = 1.0,
) -> dict[str, float]:
    """Average image metrics over corresponding projection views.

    Args:
        predicted_drrs: Predicted projections.
        target_drrs: Reference projections.
        data_range: Projection intensity range.

    Returns:
        Mean reprojection ``psnr`` and ``ssim``.
    """
    if not predicted_drrs or len(predicted_drrs) != len(target_drrs):
        raise ValueError("Projection sequences must have the same non-zero length.")
    psnr_values = [
        peak_signal_noise_ratio(prediction, target, data_range=data_range)
        for prediction, target in zip(predicted_drrs, target_drrs, strict=True)
    ]
    ssim_values = [
        structural_similarity(prediction, target, data_range=data_range)
        for prediction, target in zip(predicted_drrs, target_drrs, strict=True)
    ]
    return {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }
