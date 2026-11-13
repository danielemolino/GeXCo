"""Geometry-aware ray-lifting correctness tests."""

import torch

from gexco.reconstruction.geometry import ViewMetadata
from gexco.reconstruction.lifting import lift_view_features


def test_constant_feature_lifts_to_constant(
    dummy_latent_grid: torch.Tensor,
    dummy_view_metadata: list[ViewMetadata],
) -> None:
    """A constant detector feature remains constant inside the field of view."""
    feature = torch.full((1, 4, 128, 128), 2.5)
    lifted = lift_view_features(feature, dummy_latent_grid, dummy_view_metadata[0])
    assert lifted.shape == (1, 4, 32, 32, 32)
    assert torch.allclose(lifted, torch.full_like(lifted, 2.5), atol=1e-5)


def test_view_angle_changes_lifting_pattern(
    dummy_latent_grid: torch.Tensor,
    dummy_view_metadata: list[ViewMetadata],
) -> None:
    """Frontal and lateral geometry produce different lifted feature patterns."""
    horizontal = torch.linspace(0, 1, 128).view(1, 1, 1, 128).expand(1, 1, 128, 128)
    frontal = lift_view_features(horizontal, dummy_latent_grid, dummy_view_metadata[0])
    lateral = lift_view_features(horizontal, dummy_latent_grid, dummy_view_metadata[1])
    assert not torch.allclose(frontal, lateral)
