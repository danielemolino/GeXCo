"""Single-sample and manifest-batch inference for the released GeXCo checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gexco.config import default_config, load_config
from gexco.reconstruction.configuration import checkpoint_configuration

from gexco.pretraining.data import load_grayscale_image
from gexco.data.preprocessing import denormalize_hu
from gexco.reconstruction.autoencoder import (
    decode_maisi_latent,
    load_maisi_autoencoder,
)
from gexco.reconstruction.denoiser import (
    GeXCoReconstructionModel,
    load_reconstruction_checkpoint as restore_model,
)
from gexco.reconstruction.geometry import build_latent_grid, standard_biplanar_views
from gexco.reconstruction.rectified_flow import RectifiedFlowScheduler
from gexco.utils.tensors import seed_everything


def load_reconstruction_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    config_path: str | Path | None = None,
) -> tuple[GeXCoReconstructionModel, dict[str, Any], dict[str, Any]]:
    """Restore a released checkpoint and its public YAML configuration.

    Args:
        checkpoint_path: Path to a reconstruction checkpoint.
        device: Inference device.
        config_path: Optional compatible YAML; otherwise use the saved checkpoint configuration.

    Returns:
        Model, checkpoint metadata, and parsed configuration.
    """
    model, payload = restore_model(checkpoint_path, map_location=device)
    config = checkpoint_configuration(payload, config_path)
    return model.to(device).eval(), payload, config


@torch.inference_mode()
def reconstruct_latent(
    images: torch.Tensor,
    model: GeXCoReconstructionModel,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Reconstruct an unscaled MAISI latent from biplanar DRRs.

    Args:
        images: Normalized tensor shaped ``(B,2,1,128,128)``.
        model: Restored GeXCo model.
        checkpoint: Serialized checkpoint metadata.
        config: Reconstruction configuration.
        seed: Gaussian-noise seed.

    Returns:
        Unscaled latent shaped ``(B,4,32,32,32)``.
    """
    seed_everything(seed)
    device = next(model.parameters()).device
    images = images.to(device=device, dtype=torch.float32)
    latent_size = int(config["data"]["volume_size"]) // int(config["vae"]["spatial_downsample"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn(
        (
            images.shape[0],
            int(config["vae"]["latent_channels"]),
            latent_size,
            latent_size,
            latent_size,
        ),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    spacing = (
        torch.tensor(
            config["data"]["voxel_spacing_mm"],
            device=device,
            dtype=sample.dtype,
        )
        .unsqueeze(0)
        .expand(images.shape[0], -1)
    )
    grid = build_latent_grid(
        sample.shape[-3:],
        sample.shape[0],
        spacing_zyx_mm=spacing,
        downsample_factor=int(config["vae"]["spatial_downsample"]),
        device=device,
        dtype=sample.dtype,
    )
    views = standard_biplanar_views(
        height=int(config["drr"]["detector_size"]),
        width=int(config["drr"]["detector_size"]),
        sdd=float(config["drr"]["source_to_detector_mm"]),
        sid=float(config["drr"]["source_to_patient_mm"]),
        pixel_size_mm=float(config["drr"]["effective_pixel_size_mm"]),
        detector_axis_convention=config["drr"].get("detector_axis_convention", "standard"),
    )
    flow_config = config["rectified_flow"]
    flow = RectifiedFlowScheduler(
        num_train_timesteps=int(flow_config["num_train_timesteps"]),
        use_discrete_timesteps=bool(flow_config["use_discrete_timesteps"]),
        use_timestep_transform=bool(flow_config["use_timestep_transform"]),
        transform_scale=float(flow_config.get("transform_scale", 1.0)),
    )
    flow.set_timesteps(
        int(config["inference"]["ode_steps"]),
        device=device,
        input_img_size_numel=latent_size**3,
    )
    # MAISI sampling uses a float16 schedule and integer network timesteps.
    flow.timesteps = flow.timesteps.half()
    guidance = float(config["inference"]["guidance_scale"])
    zero_images = torch.zeros_like(images)
    cache = bool(config["inference"].get("conditioning_cache", True))
    conditional_features = model.conditioner(images, views, grid) if cache else None
    unconditional_features = model.conditioner(zero_images, views, grid) if cache and guidance != 1.0 else None
    for index, timestep in enumerate(flow.timesteps):
        next_timestep = int(flow.timesteps[index + 1]) if index + 1 < len(flow.timesteps) else 0.0
        batch_timestep = torch.full((images.shape[0],), int(timestep.item()), device=device, dtype=flow.timesteps.dtype)
        if cache:
            conditional = model.denoiser.predict_velocity(
                sample, batch_timestep, conditional_features, spacing_tensor=spacing * 100.0
            )
        else:
            conditional = model(sample, batch_timestep, images, views, grid, spacing_tensor=spacing * 100.0)
        if guidance == 1.0:
            velocity = conditional
        else:
            if cache:
                unconditional = model.denoiser.predict_velocity(
                    sample, batch_timestep, unconditional_features, spacing_tensor=spacing * 100.0
                )
            else:
                unconditional = model(sample, batch_timestep, zero_images, views, grid, spacing_tensor=spacing * 100.0)
            velocity = unconditional + guidance * (conditional - unconditional)
        sample, _ = flow.step(
            velocity,
            int(timestep.item()),
            sample,
            next_timestep=next_timestep,
        )
    scale_factor = float(checkpoint["scale_factor"])
    return sample / scale_factor


def decode_with_vae(latent: torch.Tensor, vae_checkpoint: str | Path) -> torch.Tensor:
    """Decode a latent using a native or TorchScript MAISI checkpoint.

    Args:
        latent: Unscaled four-channel MAISI latent.
        vae_checkpoint: Native ``autoencoder_epoch273.pt`` or TorchScript decoder.

    Returns:
        Reconstructed CT volume.
    """
    try:
        vae = torch.jit.load(str(vae_checkpoint), map_location=latent.device).eval()
    except RuntimeError:
        autoencoder = load_maisi_autoencoder(
            vae_checkpoint,
            device=latent.device,
        )
        return decode_maisi_latent(latent, autoencoder)
    decoded = vae.decode(latent) if hasattr(vae, "decode") else vae(latent)
    return decoded.float().clamp(0.0, 1.0)


decode_with_torchscript = decode_with_vae


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read a JSON or JSONL inference manifest."""
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    records = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Manifest must contain a sample list.")
    return records


def load_input_image(path, image_size, *, input_kind="drr", stats_path=None, normalization="dataset_percentile_fixed",
                     view=None, radiograph_config=None):
    if input_kind == "drr":
        return load_grayscale_image(path, image_size, stats_path=stats_path, normalization=normalization)
    if input_kind == "normalized":
        return load_grayscale_image(path, image_size, normalization="none")
    if input_kind != "acquired":
        raise ValueError(f"Unknown input kind: {input_kind}")
    from gexco.data.radiographs import load_acquired

    return load_acquired(path, image_size, view=view, config=radiograph_config)


def main() -> None:
    """Parse CLI inputs and reconstruct one or more samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Compatible YAML override; defaults to checkpoint settings, then packaged settings if absent",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--frontal", type=Path)
    parser.add_argument("--lateral", type=Path)
    parser.add_argument("--case-id", help="Output case ID for a single frontal/lateral pair")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--input-kind", choices=["drr", "acquired", "normalized"], default="drr")
    parser.add_argument("--radiograph-config", type=Path, default=default_config("radiographs"),
                        help="DICOM preprocessing calibration for acquired inputs")
    parser.add_argument("--ode-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int, default=None)
    arguments = parser.parse_args()
    if arguments.frontal and not arguments.lateral:
        parser.error("--lateral is required with --frontal.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, config = load_reconstruction_checkpoint(
        arguments.checkpoint,
        device,
        arguments.config,
    )
    if arguments.seed is None:
        arguments.seed = int(config["inference"].get("seed", 0))
    if arguments.ode_steps is not None:
        config["inference"]["ode_steps"] = arguments.ode_steps
    if arguments.guidance_scale is not None:
        config["inference"]["guidance_scale"] = arguments.guidance_scale
    autoencoder = load_maisi_autoencoder(arguments.vae_checkpoint, device=device)
    radiograph_config = load_config(arguments.radiograph_config) if arguments.input_kind == "acquired" else None
    records = (
        _read_manifest(arguments.manifest)
        if arguments.manifest
        else [
            {
                "case_id": arguments.case_id or arguments.frontal.stem.removesuffix("_F"),
                "views": [
                    {"path": str(arguments.frontal), "angle_deg": 0.0},
                    {"path": str(arguments.lateral), "angle_deg": 90.0},
                ],
            }
        ]
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    seed_policy = config["inference"].get("seed_policy", "per_case_fixed")
    if seed_policy not in {"per_case_fixed", "per_case_index"}:
        raise ValueError("seed_policy must be per_case_fixed or per_case_index")
    for index, record in enumerate(records):
        case_seed = int(record.get("seed", arguments.seed + (index if seed_policy == "per_case_index" else 0)))
        ordered = sorted(record["views"], key=lambda item: float(item["angle_deg"]))
        if [float(item["angle_deg"]) for item in ordered] != [0.0, 90.0]:
            raise ValueError("Inference needs exactly the frontal 0-degree and lateral 90-degree views")
        images = torch.stack(
            [
                load_input_image(
                    item["path"],
                    int(config["encoder"]["input_size"]),
                    input_kind=arguments.input_kind,
                    view="frontal" if float(item["angle_deg"]) == 0 else "lateral",
                    radiograph_config=radiograph_config,
                    stats_path=config["drr"].get("stats_path"),
                    normalization=str(
                        config["drr"].get(
                            "normalization",
                            "dataset_percentile_fixed",
                        )
                    ),
                )
                for item in ordered
            ]
        ).unsqueeze(0)
        latent = reconstruct_latent(
            images,
            model,
            checkpoint,
            config,
            seed=case_seed,
        )
        case_id = str(record.get("case_id", index))
        if Path(case_id).name != case_id or case_id in {"", ".", ".."}:
            raise ValueError("case_id must be a plain filename")
        volume = decode_maisi_latent(latent, autoencoder)
        volume_hu = denormalize_hu(volume, tuple(config["data"]["hu_clip"]))
        from gexco.data.io import save_ct

        affine = np.asarray(record["affine"]) if "affine" in record else None
        save_ct(arguments.output_dir / f"{case_id}.nii.gz", volume_hu[0, 0].cpu().numpy(), affine)
        print(f"Saved {case_id}.nii.gz: CT128, 3 mm, int16 HU; seed={case_seed}")


if __name__ == "__main__":
    main()
