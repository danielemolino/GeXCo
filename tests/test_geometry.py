"""Known-point projection and back-projection tests."""

import torch

from gexco.reconstruction.geometry import (
    ViewMetadata,
    build_latent_grid,
    detector_pixels_to_rays,
    project_points_to_detector,
    standard_biplanar_views,
)


def test_world_origin_projects_to_detector_centre(
    dummy_view_metadata: list[ViewMetadata],
) -> None:
    """The isocentre projects to the principal point in a frontal view."""
    view = dummy_view_metadata[0]
    point = torch.zeros(1, 3)
    pixel = project_points_to_detector(point, view)
    expected = torch.tensor([[(view.width - 1) / 2, (view.height - 1) / 2]])
    assert torch.allclose(pixel, expected)


def test_projection_ray_round_trip(dummy_view_metadata: list[ViewMetadata]) -> None:
    """Any point along a back-projected ray returns to its detector pixel."""
    view = dummy_view_metadata[0]
    pixel = torch.tensor([[31.25, 78.5]])
    origin, direction = detector_pixels_to_rays(pixel, view)
    point = origin + 1100.0 * direction
    recovered = project_points_to_detector(point, view)
    assert torch.allclose(recovered, pixel, atol=1e-4)


def test_projection_uses_configured_detector_coordinate_frame() -> None:
    """Asymmetric 3D landmarks project with the configured detector axes."""
    view = standard_biplanar_views()[0]
    point = torch.tensor([[23.0, -17.0, 11.0]])
    pixel = project_points_to_detector(point, view)
    expected = torch.tensor([[56.2326, 60.0243]])
    assert torch.allclose(pixel, expected, atol=1e-3)


def test_physical_hdf5_ray_round_trip() -> None:
    """Back-projection applies the inverse detector coordinate transform."""
    view = standard_biplanar_views()[0]
    pixel = torch.tensor([[31.25, 78.5]])
    origin, direction = detector_pixels_to_rays(pixel, view)
    recovered = project_points_to_detector(origin + 1100.0 * direction, view)
    assert torch.allclose(recovered, pixel, atol=1e-4)


def test_points_behind_source_are_invalid(dummy_view_metadata: list[ViewMetadata]) -> None:
    """Points behind the source plane produce NaN detector coordinates."""
    view = dummy_view_metadata[0]
    centre = torch.tensor([[(view.width - 1) / 2, (view.height - 1) / 2]])
    origin, direction = detector_pixels_to_rays(centre, view)
    projected = project_points_to_detector(origin - 100.0 * direction, view)
    assert torch.isnan(projected).all()


def test_latent_spacing_maps_canonical_axes_to_tensor_dimensions() -> None:
    """Canonical x/y/z spacing follows latent D/H/W, including anisotropic scans."""
    grid = build_latent_grid(
        (3, 4, 5),
        spacing_xyz_mm=torch.tensor([1.0, 2.0, 3.0]),
        downsample_factor=1,
    )
    assert torch.equal(grid[0, 0, 2, 2], torch.tensor([-1.0, 1.0, 0.0]))
    assert torch.equal(grid[0, 2, 2, 2], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(grid[0, 1, 0, 2], torch.tensor([0.0, -3.0, 0.0]))
    assert torch.equal(grid[0, 1, 3, 2], torch.tensor([0.0, 3.0, 0.0]))
    assert torch.equal(grid[0, 1, 2, 0], torch.tensor([0.0, 1.0, -6.0]))
    assert torch.equal(grid[0, 1, 2, 4], torch.tensor([0.0, 1.0, 6.0]))
