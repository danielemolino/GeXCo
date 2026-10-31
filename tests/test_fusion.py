"""Average and adaptive per-voxel fusion tests."""

import torch

from gexco.reconstruction.fusion import AdaptiveViewFusion, AverageViewFusion


def test_average_fusion_identical_inputs() -> None:
    """Averaging identical lifted views is the identity operation."""
    view = torch.randn(2, 8, 4, 5, 6)
    fused = AverageViewFusion()([view, view.clone()])
    assert torch.allclose(fused, view)


def test_weighted_fusion_shape_and_normalized_weights() -> None:
    """Adaptive fusion preserves spatial shape and normalizes across views."""
    views = [torch.randn(2, 8, 4, 5, 6) for _ in range(3)]
    fusion = AdaptiveViewFusion(8, 12)
    fused, weights = fusion(views, return_weights=True)
    assert fused.shape == (2, 12, 4, 5, 6)
    assert weights.shape == (2, 3, 1, 4, 5, 6)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-6)
