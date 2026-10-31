"""Architecture-level checks for the released checkpoint contracts."""

from pathlib import Path

import pytest
import torch

from gexco.pretraining.models import build_world_model
from gexco.reconstruction.denoiser import Denoiser3D
from gexco.reconstruction.encoder import PretrainedViewEncoder


def test_pretraining_component_checkpoint_round_trip(tmp_path: Path) -> None:
    """A development-format world-model checkpoint restores without key remapping."""
    model = build_world_model(
        input_size=32,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        predictor_depth=1,
        predictor_num_heads=4,
        projector_hidden_dim=64,
        projector_dim=32,
    )
    checkpoint = tmp_path / "world_model.pt"
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "view_predictor": model.predictor.state_dict(),
            "enc_projector": model.enc_projector.state_dict(),
            "pred_projector": model.pred_projector.state_dict(),
            "config": {
                "input_size": 32,
                "in_channels": 1,
                "patch_size": 8,
                "embed_dim": 32,
                "depth": 2,
                "num_heads": 4,
                "predictor_embed_dim": 32,
                "predictor_depth": 1,
                "predictor_num_heads": 4,
                "use_projector": True,
                "proj_hidden_dim": 64,
                "proj_dim": 32,
                "pose_scale_learnable": True,
                "no_encoder_pose": False,
            },
        },
        checkpoint,
    )
    encoder = PretrainedViewEncoder(
        checkpoint,
        use_enc_projector=True,
        trainable=False,
    )
    output = encoder(torch.randn(2, 1, 32, 32), torch.tensor([0.0, torch.pi / 2]))
    assert output.shape == (2, 16, 32)


def test_maisi_denoiser_has_132_channel_checkpoint_stem() -> None:
    """Even a reduced CPU fixture preserves the paper's 132-channel stem."""
    try:
        denoiser = Denoiser3D(
            channels=(32, 64),
            attention_levels=(1,),
            res_blocks_per_level=1,
        )
    except ImportError:
        pytest.skip("MONAI is not installed.")
    assert denoiser.state_dict()["conv_in.conv.weight"].shape[1] == 132
