"""Known-point projection and back-projection tests."""

import torch

from gexco.reconstruction.geometry import (
    ViewMetadata,
    detector_pixels_to_rays,
    project_points_to_detector,
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


def test_points_behind_source_are_invalid(dummy_view_metadata: list[ViewMetadata]) -> None:
    """Points behind the source plane produce NaN detector coordinates."""
    view = dummy_view_metadata[0]
    centre = torch.tensor([[(view.width - 1) / 2, (view.height - 1) / 2]])
    origin, direction = detector_pixels_to_rays(centre, view)
    projected = project_points_to_detector(origin - 100.0 * direction, view)
    assert torch.isnan(projected).all()
