"""Shared synthetic CT, latent, DRR, and geometry fixtures."""

from __future__ import annotations

import pytest
import torch

from gexco.reconstruction.geometry import ViewMetadata, build_latent_grid, standard_biplanar_views


@pytest.fixture(scope="session")
def dummy_volume() -> torch.Tensor:
    """Return a synthetic ``(1,1,128,128,128)`` CT volume."""
    generator = torch.Generator().manual_seed(1)
    return torch.rand((1, 1, 128, 128, 128), generator=generator)


@pytest.fixture(scope="session")
def dummy_latent() -> torch.Tensor:
    """Return a synthetic ``(1,4,32,32,32)`` MAISI latent."""
    generator = torch.Generator().manual_seed(2)
    return torch.rand((1, 4, 32, 32, 32), generator=generator)


@pytest.fixture(scope="session")
def dummy_drr_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Return synthetic frontal and lateral ``128x128`` DRRs."""
    generator = torch.Generator().manual_seed(3)
    return (
        torch.rand((1, 1, 128, 128), generator=generator),
        torch.rand((1, 1, 128, 128), generator=generator),
    )


@pytest.fixture(scope="session")
def dummy_view_metadata() -> list[ViewMetadata]:
    """Return standard frontal/lateral DiffDRR-style geometry."""
    return standard_biplanar_views(
        height=128,
        width=128,
        sdd=1500.0,
        sid=972.0,
        pixel_size_mm=4.8,
    )


@pytest.fixture(scope="session")
def dummy_latent_grid() -> torch.Tensor:
    """Return a ``(1,32,32,32,3)`` world-coordinate grid."""
    return build_latent_grid((32, 32, 32), extent_mm=(80.0, 80.0, 80.0))
