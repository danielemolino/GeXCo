"""Checkpoint-exact MAISI autoencoder used to encode and decode CT latents."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

try:
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
        AutoencoderKlMaisi,
    )
except ImportError:  # pragma: no cover - optional dependency error path
    AutoencoderKlMaisi = None  # type: ignore[assignment,misc]


def build_maisi_autoencoder(*, norm_float16: bool = True) -> nn.Module:
    """Build the frozen MAISI VAE architecture used by GeXCo.

    Returns:
        MAISI autoencoder with a four-channel, 4x-downsampled latent space.
    """
    if AutoencoderKlMaisi is None:
        raise ImportError("MONAI is required to construct the MAISI autoencoder.")
    return AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=(64, 128, 256),
        num_res_blocks=(2, 2, 2),
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=norm_float16,
        num_splits=4,
        dim_split=1,
    )


def load_maisi_autoencoder(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Restore the native MAISI VAE state dict with strict key checking.

    Args:
        checkpoint_path: Path to ``autoencoder_epoch273.pt``.
        device: Target inference device.

    Returns:
        Frozen autoencoder in evaluation mode.
    """
    target = torch.device(device)
    autoencoder = build_maisi_autoencoder(norm_float16=target.type == "cuda")
    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("MAISI checkpoint must contain a state dictionary.")
    for key in ("generator_state_dict", "autoencoder_state_dict", "state_dict"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            payload = nested
            break
    autoencoder.load_state_dict(payload, strict=True, assign=True)
    autoencoder = autoencoder.to(device=target, dtype=torch.float32)
    if target.type == "cuda":
        autoencoder = autoencoder.half()
    return autoencoder.eval().requires_grad_(False)


@torch.inference_mode()
def decode_maisi_latent(latent: torch.Tensor, autoencoder: nn.Module) -> torch.Tensor:
    """Decode an unscaled MAISI latent into a normalized CT volume.

    Args:
        latent: Tensor shaped ``(B,4,D,H,W)``.
        autoencoder: Restored MAISI autoencoder.

    Returns:
        Decoded tensor shaped ``(B,1,4D,4H,4W)``.
    """
    parameter = next(autoencoder.parameters())
    decoded = autoencoder.decode_stage_2_outputs(  # type: ignore[attr-defined]
        latent.to(device=parameter.device, dtype=parameter.dtype)
    )
    return decoded.float().clamp(0.0, 1.0)


__all__ = [
    "build_maisi_autoencoder",
    "decode_maisi_latent",
    "load_maisi_autoencoder",
]
