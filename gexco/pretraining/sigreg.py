# SIGReg adapted from LeWorldModel; Copyright (c) 2026 Lucas Maes.
# MIT license: see THIRD_PARTY_NOTICES.md.
"""Checkpoint-compatible SIGReg regularizer used by GeXCo pretraining."""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Measure departure of random embedding projections from a unit Gaussian."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        """Initialize the Epps-Pulley sketch.

        Args:
            knots: Number of characteristic-function evaluation points.
            num_proj: Number of random one-dimensional projections.
        """
        super().__init__()
        if knots < 2 or num_proj < 1:
            raise ValueError("knots must be >=2 and num_proj must be positive.")
        self.num_proj = int(num_proj)
        points = torch.linspace(0, 3, knots, dtype=torch.float32)
        spacing = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * spacing, dtype=torch.float32)
        weights[[0, -1]] = spacing
        gaussian_cf = torch.exp(-points.square() / 2.0)
        # Buffer names intentionally match the serialized checkpoint.
        self.register_buffer("t", points)
        self.register_buffer("phi", gaussian_cf)
        self.register_buffer("weights", weights * gaussian_cf)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            embeddings: Tensor shaped ``(T,B,D)`` or ``(B,D)``.

        Returns:
            Non-negative scalar regularization loss.
        """
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (B,D) or (T,B,D).")
        projections = torch.randn(
            embeddings.shape[-1],
            self.num_proj,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        projections = projections.div_(projections.norm(p=2, dim=0))
        phases = (embeddings @ projections).unsqueeze(-1) * self.t.to(dtype=embeddings.dtype)
        error = (phases.cos().mean(-3) - self.phi.to(dtype=embeddings.dtype)).square() + phases.sin().mean(-3).square()
        statistic = (error @ self.weights.to(dtype=embeddings.dtype)) * embeddings.size(-2)
        return statistic.mean()


def build_sigreg(knots: int = 17, num_proj: int = 1024) -> SIGReg:
    """Build the paper SIGReg regularizer.

    Args:
        knots: Number of characteristic-function knots.
        num_proj: Number of random projections.

    Returns:
        Configured :class:`SIGReg`.
    """
    return SIGReg(knots=knots, num_proj=num_proj)
