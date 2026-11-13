"""MAISI 3D U-Net and checkpoint-compatible GeXCo reconstruction wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from gexco.reconstruction.conditioning import GeometryConditioner
from gexco.reconstruction.encoder import PretrainedViewEncoder
from gexco.reconstruction.geometry import ViewMetadata
from gexco.config import default_config, load_config
from gexco.reconstruction.configuration import validate_architecture

try:
    from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
        DiffusionModelUNetMaisi,
    )
except ImportError:  # pragma: no cover - optional dependency error path
    DiffusionModelUNetMaisi = None  # type: ignore[assignment,misc]


class Denoiser3D(
    DiffusionModelUNetMaisi if DiffusionModelUNetMaisi is not None else nn.Module  # type: ignore[misc]
):
    """Exact MAISI denoiser used by the reconstruction checkpoint."""

    def __init__(
        self,
        latent_channels: int = 4,
        conditioning_channels: int = 128,
        channels: Sequence[int] = (64, 128, 256, 512),
        attention_levels: Sequence[int] = (2, 3),
        attention_head_channels: int = 32,
        res_blocks_per_level: int = 2,
        num_class_embeds: int = 128,
        include_spacing_input: bool = True,
    ) -> None:
        """Initialize the checkpoint-exact MAISI U-Net.

        Args:
            latent_channels: Noisy latent channels.
            conditioning_channels: Concatenated GeXCo channels.
            channels: U-Net widths.
            attention_levels: Zero-based levels with self-attention.
            attention_head_channels: Channels assigned to each attention head.
            res_blocks_per_level: Residual blocks per level.
            num_class_embeds: MAISI modality embedding count.
            include_spacing_input: Inject physical voxel spacing.
        """
        if DiffusionModelUNetMaisi is None:
            raise ImportError("MONAI is required for the MAISI denoiser; install the pinned GeXCo dependencies.")
        widths = tuple(int(value) for value in channels)
        enabled = {int(value) for value in attention_levels}
        attention_mask = tuple(index in enabled for index in range(len(widths)))
        head_channels = tuple(attention_head_channels if active else 0 for active in attention_mask)
        super().__init__(
            spatial_dims=3,
            in_channels=int(latent_channels + conditioning_channels),
            out_channels=int(latent_channels),
            num_res_blocks=int(res_blocks_per_level),
            num_channels=widths,
            attention_levels=attention_mask,
            num_head_channels=head_channels,
            with_conditioning=False,
            cross_attention_dim=None,
            use_flash_attention=False,
            num_class_embeds=int(num_class_embeds),
            resblock_updown=True,
            include_fc=True,
            include_top_region_index_input=False,
            include_bottom_region_index_input=False,
            include_spacing_input=bool(include_spacing_input),
        )
        self.latent_channels = int(latent_channels)
        self.conditioning_channels = int(conditioning_channels)

    def predict_velocity(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        spacing_tensor: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict rectified-flow velocity from latent and spatial condition.

        Args:
            latent: Tensor shaped ``(B,4,D,H,W)``.
            timesteps: MAISI timesteps shaped ``(B,)``.
            conditioning: Tensor shaped ``(B,128,D,H,W)``.
            spacing_tensor: MAISI spacing condition, conventionally millimetres times 100.
            class_labels: Modality labels; CT uses ones.

        Returns:
            Velocity tensor shaped like ``latent``.
        """
        if spacing_tensor is None:
            spacing_tensor = torch.full(
                (latent.shape[0], 3),
                100.0,
                device=latent.device,
                dtype=latent.dtype,
            )
        if class_labels is None:
            class_labels = torch.ones(
                latent.shape[0],
                device=latent.device,
                dtype=torch.long,
            )
        return super().forward(
            x=torch.cat([latent, conditioning], dim=1),
            timesteps=timesteps,
            spacing_tensor=spacing_tensor,
            class_labels=class_labels,
        )


class GeXCoReconstructionModel(nn.Module):
    """Top-level module whose state keys match ``unet_state_dict`` exactly."""

    def __init__(self, denoiser: Denoiser3D, conditioner: GeometryConditioner) -> None:
        """Initialize the reconstruction model.

        Args:
            denoiser: Exact MAISI velocity network.
            conditioner: Geometry-aware view conditioner.
        """
        super().__init__()
        self.denoiser = denoiser
        self.conditioner = conditioner

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        views: Sequence[ViewMetadata],
        latent_grid: torch.Tensor,
        *,
        spacing_tensor: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        return_conditioning: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict velocity conditioned on biplanar radiographs.

        Args:
            noisy_latent: Tensor shaped ``(B,4,D,H,W)``.
            timesteps: MAISI timesteps shaped ``(B,)``.
            images: Tensor shaped ``(B,V,1,128,128)``.
            views: Matching projection metadata.
            latent_grid: World-coordinate latent grid.
            spacing_tensor: MAISI spacing condition.
            class_labels: MAISI modality labels.
            return_conditioning: Also return the fused condition.

        Returns:
            Predicted velocity, optionally paired with its condition.
        """
        conditioning = self.conditioner(images, views, latent_grid)
        if not isinstance(conditioning, torch.Tensor):
            raise RuntimeError("Conditioner returned unexpected details.")
        velocity = self.denoiser.predict_velocity(
            noisy_latent,
            timesteps,
            conditioning,
            spacing_tensor=spacing_tensor,
            class_labels=class_labels,
        )
        return (velocity, conditioning) if return_conditioning else velocity


def build_denoiser(
    latent_channels: int = 4,
    conditioning_channels: int = 128,
    channels: Sequence[int] = (64, 128, 256, 512),
    attention_levels: Sequence[int] = (2, 3),
    attention_head_channels: int = 32,
    res_blocks_per_level: int = 2,
) -> Denoiser3D:
    """Build the exact MAISI denoiser used in the paper.

    Args:
        latent_channels: Noisy MAISI latent channels.
        conditioning_channels: Concatenated GeXCo condition channels.
        channels: U-Net channel widths.
        attention_levels: Zero-based levels with self-attention.
        attention_head_channels: Channels assigned to each attention head.
        res_blocks_per_level: Residual blocks per U-Net level.

    Returns:
        Checkpoint-compatible MAISI denoiser.
    """
    return Denoiser3D(
        latent_channels=latent_channels,
        conditioning_channels=conditioning_channels,
        channels=channels,
        attention_levels=attention_levels,
        attention_head_channels=attention_head_channels,
        res_blocks_per_level=res_blocks_per_level,
    )


def build_reconstruction_model(
    *,
    pretraining_checkpoint: str | Path | None = None,
    encoder_trainable: bool = True,
) -> GeXCoReconstructionModel:
    """Build the checkpoint-compatible full reconstruction architecture.

    Args:
        pretraining_checkpoint: Optional world-model ``best.pt`` initialization.
        encoder_trainable: Fine-tune the encoder and its projector.

    Returns:
        Full GeXCo reconstruction model.
    """
    encoder = PretrainedViewEncoder(
        checkpoint_path=pretraining_checkpoint,
        use_enc_projector=True,
        trainable=encoder_trainable,
        require_reconstruction_architecture=True,
    )
    conditioner = GeometryConditioner(
        encoder=encoder,
        encoder_channels=128,
        fusion_channels=128,
        latent_channels=4,
        use_adaptive_fusion=True,
    )
    return GeXCoReconstructionModel(build_denoiser(), conditioner)


def _normalize_reconstruction_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    expected_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Normalize serialized encoder module names.

    The released weights predate the public package naming. The translation is
    structural: it changes only an otherwise-unrecognized container directly
    below ``conditioner.encoder`` when the resulting key is expected by the
    current model.

    Args:
        state_dict: Serialized reconstruction weights.
        expected_keys: State keys exposed by the current GeXCo model.

    Returns:
        State dictionary using the public GeXCo module names.
    """
    encoder_prefix = "conditioner.encoder."
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key
        if key not in expected_keys and key.startswith(encoder_prefix):
            suffix = key[len(encoder_prefix) :]
            _, separator, remainder = suffix.partition(".")
            candidate = f"{encoder_prefix}view_encoder.{remainder}"
            if separator and candidate in expected_keys:
                normalized_key = candidate
        if normalized_key in normalized:
            raise RuntimeError(f"Duplicate state key after normalization: {normalized_key}")
        normalized[normalized_key] = value
    return normalized


def load_reconstruction_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[GeXCoReconstructionModel, dict[str, Any]]:
    """Restore a reconstruction checkpoint with strict key checking.

    Args:
        checkpoint_path: Reconstruction checkpoint path.
        map_location: PyTorch load target.

    Returns:
        Restored model and checkpoint metadata.
    """
    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    validate_architecture(payload.get("config") or load_config(default_config("reconstruction")))
    with torch.device("meta"):
        model = build_reconstruction_model(pretraining_checkpoint=None)
    expected_keys = set(model.state_dict())
    state_dict = _normalize_reconstruction_state_dict(
        payload["unet_state_dict"],
        expected_keys,
    )
    model.load_state_dict(state_dict, strict=True, assign=True)
    target = torch.device(map_location)
    if target.type != "cpu":
        model = model.to(target)
    return model, payload


__all__ = [
    "Denoiser3D",
    "GeXCoReconstructionModel",
    "build_denoiser",
    "build_reconstruction_model",
    "load_reconstruction_checkpoint",
]
