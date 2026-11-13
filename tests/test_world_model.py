"""Angular predictor and multi-target world-model tests."""

import torch

from gexco.pretraining.models import (
    ViewPredictor,
    WorldModel,
    WorldModelEncoder,
    build_world_model,
)


def test_predictor_single_target_shape() -> None:
    """The angular predictor preserves anchor token shape."""
    predictor = ViewPredictor(
        num_patches=16,
        input_dim=32,
        predictor_embed_dim=32,
        depth=2,
        num_heads=4,
    )
    tokens = torch.randn(2, 16, 32)
    output = predictor(tokens, torch.tensor([0.1, -0.2]))
    assert output.shape == tokens.shape


def test_world_model_eight_target_shape() -> None:
    """The paper's ``K=8`` targets produce an explicit target dimension."""
    encoder = WorldModelEncoder(
        input_size=32,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
    )
    model = WorldModel(
        encoder,
        ViewPredictor(
            num_patches=16,
            input_dim=32,
            predictor_embed_dim=32,
            depth=1,
            num_heads=4,
        ),
    )
    anchor = torch.randn(2, 1, 32, 32)
    angular_deltas = torch.randn(2, 8)
    output = model(anchor, angular_deltas)
    assert output.shape == (2, 8, 16, 32)


def test_default_checkpoint_parameter_count_and_state_names() -> None:
    """The public model exactly matches the released pretraining state contract."""
    model = build_world_model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_609_859
    assert model.encoder.state_dict()["encoder.patch_embed.weight"].shape == (128, 1, 8, 8)
    assert model.enc_projector is not None
    assert model.enc_projector.state_dict()["net.0.weight"].shape == (512, 128)
