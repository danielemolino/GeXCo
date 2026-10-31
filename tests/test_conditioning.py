"""End-to-end encoder, lifting, and fusion pipeline tests."""

import pytest
import torch

from gexco.reconstruction.conditioning import GeometryConditioner
from gexco.reconstruction.encoder import ViTEncoder
from gexco.reconstruction.geometry import ViewMetadata


@pytest.mark.parametrize("num_views", [1, 2])
def test_full_conditioning_shape(
    num_views: int,
    dummy_drr_pair: tuple[torch.Tensor, torch.Tensor],
    dummy_view_metadata: list[ViewMetadata],
    dummy_latent_grid: torch.Tensor,
) -> None:
    """Conditioning supports both single-view and biplanar inputs."""
    encoder = ViTEncoder(embed_dim=16, depth=1, num_heads=4)
    conditioner = GeometryConditioner(encoder, use_adaptive_fusion=False)
    output = conditioner(
        list(dummy_drr_pair[:num_views]),
        dummy_view_metadata[:num_views],
        dummy_latent_grid,
    )
    assert output.shape == (1, 16, 32, 32, 32)
