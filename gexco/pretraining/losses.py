"""Predictive representation losses for GeXCo world-model pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def predictive_loss(predicted_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """Compute the mean-squared JEPA prediction loss.

    Args:
        predicted_tokens: Predicted target tokens.
        target_tokens: Target tokens from the same shared encoder.

    Returns:
        Scalar mean-squared error.
    """
    if predicted_tokens.shape != target_tokens.shape:
        raise ValueError("predicted_tokens and target_tokens must have identical shapes.")
    return F.mse_loss(predicted_tokens, target_tokens)


def cosine_prediction_loss(predicted_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """Compute mean cosine distance between predicted and target tokens.

    Args:
        predicted_tokens: Predicted target tokens.
        target_tokens: Target tokens with the same shape.

    Returns:
        Scalar cosine-distance loss.
    """
    if predicted_tokens.shape != target_tokens.shape:
        raise ValueError("predicted_tokens and target_tokens must have identical shapes.")
    return 1.0 - F.cosine_similarity(predicted_tokens, target_tokens, dim=-1).mean()


# Loss alias.
loss_mim_drr = predictive_loss
