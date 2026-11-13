"""Strip resume-only state from a GeXCo training checkpoint for distribution."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from gexco.config import default_config, load_config, portable_config

GITHUB_RELEASE_ASSET_LIMIT = 2 * 1024**3


def export_inference_checkpoint(
    source: Path,
    destination: Path,
    config_path: Path | None = None,
) -> tuple[int, str]:
    """Save only the model and inference metadata, omitting optimizer/RNG state."""
    payload = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    required = {"unet_state_dict", "scale_factor", "num_train_timesteps"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Training checkpoint is missing release fields: {sorted(missing)}")

    config = load_config(config_path or default_config("reconstruction"))
    stats_digest = hashlib.sha256(Path(config["drr"]["stats_path"]).read_bytes()).hexdigest()
    config.pop("training", None)
    config.setdefault("encoder", {})["checkpoint"] = None
    config.setdefault("vae", {})["checkpoint"] = None
    config.setdefault("drr", {})["stats_path"] = "configs/drr_train_global_stats.json"
    scale_factor = float(torch.as_tensor(payload["scale_factor"]).detach().cpu().item())
    config["vae"]["scale_factor"] = scale_factor
    config["rectified_flow"]["num_train_timesteps"] = int(payload["num_train_timesteps"])
    state_dict = payload["unet_state_dict"]
    inference_only_prefixes = (
        "conditioner.encoder.view_predictor.",
        "conditioner.encoder.pred_projector.",
    )
    inference_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(inference_only_prefixes)
    }
    removed = len(state_dict) - len(inference_state_dict)
    if removed == 0:
        raise ValueError("Expected training-only Any-Anchor predictor/projector parameters were not found.")
    inference_payload = {
        "unet_state_dict": inference_state_dict,
        "config": portable_config(config),
        "drr_stats_sha256": stats_digest,
        "scale_factor": scale_factor,
        "num_train_timesteps": int(payload["num_train_timesteps"]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_payload, destination)
    size = destination.stat().st_size
    if size >= GITHUB_RELEASE_ASSET_LIMIT:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Inference checkpoint is {size} bytes; GitHub requires each release asset to be under 2 GiB."
        )
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return size, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Full resumable training checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Inference-only checkpoint destination")
    parser.add_argument("--config", type=Path, help="Reconstruction YAML; defaults to the packaged release config")
    args = parser.parse_args()
    size, digest = export_inference_checkpoint(args.input, args.output, args.config)
    print(f"Saved {args.output} ({size} bytes), SHA-256 {digest}")


if __name__ == "__main__":
    main()
