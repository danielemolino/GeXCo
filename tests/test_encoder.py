"""Shape and pose-encoding tests for the shared ViT encoder."""

import pytest
import torch

from gexco.reconstruction.encoder import ViTEncoder, sinusoidal_pose_encoding


@pytest.mark.parametrize("batch_size", [1, 3])
def test_encoder_forward_shape(batch_size: int) -> None:
    """ViT output follows ``(B,num_patches,embed_dim)``."""
    encoder = ViTEncoder(embed_dim=32, depth=2, num_heads=4)
    images = torch.randn(batch_size, 1, 128, 128)
    output = encoder(images)
    assert output.shape == (batch_size, 256, 32)


def test_pose_encoding_shape_and_angle_sensitivity() -> None:
    """Pose encodings preserve batch/feature dimensions and encode angle."""
    angles = torch.tensor([0.0, torch.pi / 2])
    encoding = sinusoidal_pose_encoding(angles, 32)
    assert encoding.shape == (2, 32)
    assert not torch.allclose(encoding[0], encoding[1])
