"""Training entry point for GeXCo latent reconstruction."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import json
import os
import random
import numpy as np
from gexco.utils.checkpoint import atomic_save
from gexco.utils.json import json_dumps
from typing import Any

import torch
from gexco.config import default_config, load_config, portable_config
from gexco.reconstruction.configuration import validate_architecture, statistics_sha256, checkpoint_configuration
from torch.utils.data import DataLoader

from gexco.data.datasets import ReconstructionDataset
from gexco.data.preprocessing import denormalize_hu
from gexco.evaluation.volume_metrics import volume_metrics
from gexco.reconstruction.autoencoder import load_maisi_autoencoder, decode_maisi_latent
from gexco.reconstruction.denoiser import (
    GeXCoReconstructionModel,
    build_reconstruction_model,
)
from gexco.reconstruction.geometry import build_latent_grid, standard_biplanar_views
from gexco.reconstruction.rectified_flow import RectifiedFlowScheduler
from gexco.reconstruction.inference import reconstruct_latent
from gexco.utils.tensors import seed_everything


def build_training_model(config: dict[str, Any]) -> GeXCoReconstructionModel:
    """Build the exact model described by a reconstruction YAML.

    Args:
        config: Parsed reconstruction configuration.

    Returns:
        Full checkpoint-compatible reconstruction module.
    """
    validate_architecture(config)
    checkpoint = config["encoder"].get("checkpoint")
    return build_reconstruction_model(
        pretraining_checkpoint=checkpoint,
        encoder_trainable=bool(config["encoder"].get("trainable", True)),
    )


def build_training_models(
    config: dict[str, Any],
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Return the conditioner and denoiser modules.

    Args:
        config: Parsed reconstruction configuration.

    Returns:
        Pair containing the geometry conditioner and MAISI denoiser.
    """
    model = build_training_model(config)
    return model.conditioner, model.denoiser


@contextmanager
def _validation_state(model: torch.nn.Module):
    """Keep monitoring independent of training RNG and module modes."""
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        for module, mode in modes:
            module.training = mode


def training_validation_indices(dataset, *, seed: int) -> list[int]:
    """Choose five fixed records from the training dataset without replacement."""
    if len(dataset) < 5:
        raise ValueError("Optional reconstruction validation requires at least five training samples")
    return random.Random(seed).sample(range(len(dataset)), 5)


@torch.inference_mode()
def validate_training_samples(model, dataset, config, autoencoder, indices) -> dict[str, float]:
    """Monitor HU PSNR/SSIM on five training records, without selecting weights."""
    if len(indices) != 5 or len(set(indices)) != 5:
        raise ValueError("Reconstruction validation requires five distinct training samples")
    scores = []
    with _validation_state(model):
        for index in indices:
            sample = dataset[index]
            latent = reconstruct_latent(
                sample["views"].unsqueeze(0), model,
                {"scale_factor": float(config["vae"]["scale_factor"])}, config,
                seed=int(config["inference"].get("seed", 0)),
            )
            prediction = denormalize_hu(decode_maisi_latent(latent, autoencoder)[0])
            reference = denormalize_hu(sample["volume"]).to(prediction.device)
            scores.append(volume_metrics(prediction, reference))
    return {metric: float(np.mean([score[metric] for score in scores])) for metric in ("psnr", "ssim")}


def train(config: dict[str, Any]) -> Path:
    """Train GeXCo using precomputed, unscaled MAISI latents.

    Args:
        config: Parsed reconstruction YAML.

    Returns:
        Path to ``unet_3d_last.pt``.
    """
    training = config["training"]
    if not training.get("manifest"):
        raise ValueError("Set training.manifest to a reconstruction manifest.")
    if not config["encoder"].get("checkpoint"):
        raise ValueError("Set encoder.checkpoint to the pretrained world-model `best.pt`.")
    seed_everything(int(training.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ReconstructionDataset(
        training["manifest"],
        image_size=int(config["encoder"]["input_size"]),
        volume_size=int(config["data"]["volume_size"]),
        drr_stats_path=config["drr"].get("stats_path"),
        drr_normalization=str(config["drr"].get("normalization", "dataset_percentile_fixed")),
    )
    validation = training.get("validation", False)
    if not isinstance(validation, bool):
        raise ValueError("training.validation must be a YAML boolean")
    validation_indices = training_validation_indices(dataset, seed=int(training.get("seed", 42))) if validation else []
    if validation and not config["vae"].get("checkpoint"):
        raise ValueError("Set vae.checkpoint when enabling reconstruction validation")
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training.get("num_workers", 8)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Training split is smaller than batch_size")
    model = build_training_model(config).to(device)
    autoencoder = None
    if validation:
        with _validation_state(model):
            autoencoder = load_maisi_autoencoder(config["vae"]["checkpoint"], device=device)
    encoder_parameters = list(model.conditioner.encoder.parameters())
    main_parameters = (
        list(model.denoiser.parameters())
        + ([] if model.conditioner.weighted_fusion is None else list(model.conditioner.weighted_fusion.parameters()))
        + list(model.conditioner.modulation.parameters())
    )
    optimizer = torch.optim.Adam(
        [
            {"params": main_parameters, "lr": float(training["lr_denoiser"])},
            {"params": encoder_parameters, "lr": float(training["lr_encoder"])},
        ],
        betas=tuple(training["betas"]),
        weight_decay=0.0,
    )
    total_steps = max(1, int(training["epochs"]) * len(loader))
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=total_steps,
        power=float(training.get("lr_polynomial_power", 2.0)),
    )
    flow_config = config["rectified_flow"]
    flow = RectifiedFlowScheduler(
        num_train_timesteps=int(flow_config["num_train_timesteps"]),
        use_discrete_timesteps=bool(flow_config["use_discrete_timesteps"]),
        use_timestep_transform=bool(flow_config["use_timestep_transform"]),
        sample_method=str(flow_config["sample_method"]),
        transform_scale=float(flow_config.get("transform_scale", 1.0)),
    )
    geometry = standard_biplanar_views(
        height=int(config["drr"]["detector_size"]),
        width=int(config["drr"]["detector_size"]),
        sdd=float(config["drr"]["source_to_detector_mm"]),
        sid=float(config["drr"]["source_to_patient_mm"]),
        pixel_size_mm=float(config["drr"]["effective_pixel_size_mm"]),
        detector_axis_convention=config["drr"].get("detector_axis_convention", "standard"),
    )
    voxel_spacing = torch.tensor(
        config["data"]["voxel_spacing_mm"],
        dtype=torch.float32,
        device=device,
    )
    scale_factor = float(config["vae"]["scale_factor"])
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if validation:
        (output_dir / "validation_samples.json").write_text(json.dumps({
            "source": "training.manifest",
            "case_ids": [str(dataset.records[index].get("case_id", index)) for index in validation_indices],
        }, indent=2) + "\n")
    start_epoch = 0
    if training.get("resume"):
        payload = torch.load(training["resume"], map_location="cpu", weights_only=False)
        checkpoint_configuration(payload, requested_config=config)
        model.load_state_dict(payload["unet_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
        start_epoch = int(payload["epoch"])
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    last_loss = float("nan")
    model.train()
    for epoch in range(start_epoch + 1, int(training["epochs"]) + 1):
        epoch_total = 0.0
        for batch in loader:
            if "latent" not in batch:
                raise ValueError("Every record must provide a precomputed MAISI `latent`.")
            latent = batch["latent"].to(device=device, dtype=torch.float32) * scale_factor
            images = batch["views"].to(device=device, dtype=torch.float32)
            keep = torch.rand(latent.shape[0], 1, 1, 1, 1, device=device) >= float(training["cfg_drop_prob"])
            images = images * keep
            spacing = (
                batch["spacing_mm"].to(device=device, dtype=torch.float32)
                if "spacing_mm" in batch
                else voxel_spacing.unsqueeze(0).expand(latent.shape[0], -1)
            )
            grid = build_latent_grid(
                latent.shape[-3:],
                latent.shape[0],
                spacing_zyx_mm=spacing,
                downsample_factor=int(config["vae"]["spatial_downsample"]),
                device=device,
                dtype=latent.dtype,
            )
            noise = torch.randn_like(latent)
            timesteps = flow.sample_timesteps(latent)
            noisy = flow.add_noise(latent, noise, timesteps)
            target_velocity = flow.velocity_target(latent, noise)
            predicted_velocity = model(
                noisy,
                timesteps,
                images,
                geometry,
                grid,
                spacing_tensor=spacing * 100.0,
            )
            if not isinstance(predicted_velocity, torch.Tensor):
                raise RuntimeError("Unexpected reconstruction model output.")
            loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite reconstruction loss")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            last_loss = float(loss.detach())
            epoch_total += last_loss
        row = {"epoch": epoch, "loss": epoch_total / len(loader)}
        if validation:
            row["validation"] = validate_training_samples(
                model, dataset, config, autoencoder, validation_indices,
            )
        checkpoint = output_dir / "unet_3d_last.pt"
        atomic_save(
            {
                "epoch": epoch,
                "loss": epoch_total / len(loader),
                "num_train_timesteps": flow.num_train_timesteps,
                "scale_factor": torch.tensor(scale_factor),
                "unet_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": portable_config(config),
                "drr_stats_sha256": statistics_sha256(config),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
            },
            checkpoint,
        )
        with (output_dir / "history.jsonl").open("a") as stream:
            stream.write(json_dumps(row) + "\n")
        print(json_dumps(row), flush=True)

    return output_dir / "unet_3d_last.pt"


def main() -> None:
    """Parse CLI arguments and launch reconstruction training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config("reconstruction"),
    )
    arguments = parser.parse_args()
    configuration = load_config(arguments.config)
    checkpoint = train(configuration)
    print(f"Saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
