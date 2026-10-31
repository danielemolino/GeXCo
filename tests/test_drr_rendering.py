"""Synthetic differentiable DRR rendering tests."""

import json
from pathlib import Path

import numpy as np
import torch

from gexco.data.drr_rendering import render_drr
from gexco.pretraining.data import load_grayscale_image


def test_uniform_volume_has_uniform_projection() -> None:
    """Line integrals through a uniform cube are spatially uniform at 0°."""
    volume = torch.ones(1, 1, 16, 24, 20)
    projection = render_drr(volume, angle_deg=0.0)
    assert projection.shape == (1, 1, 24, 20)
    assert torch.allclose(projection, torch.full_like(projection, 16.0))


def test_drr_shape_matches_volume_detector(dummy_volume: torch.Tensor) -> None:
    """Rendered projection height and width match the input volume."""
    projection = render_drr(dummy_volume, angle_deg=0.0, normalize=True)
    assert projection.shape == (1, 1, 128, 128)


def test_fixed_dataset_percentile_normalization(tmp_path: Path) -> None:
    """DRRs use the checkpoint's fixed training-set percentile contract."""
    image_path = tmp_path / "drr.npy"
    stats_path = tmp_path / "stats.json"
    np.save(image_path, np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    stats_path.write_text(json.dumps({"train_global": {"p1": 0.0, "p99": 2.0}}))
    image = load_grayscale_image(
        image_path,
        image_size=2,
        stats_path=stats_path,
        normalization="dataset_percentile_fixed",
    )
    expected = torch.tensor([[[0.0, 0.5], [1.0, 1.0]]])
    assert torch.allclose(image, expected)
