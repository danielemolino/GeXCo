"""Validate the fixed paper architecture and checkpoint input configuration."""

from collections.abc import Mapping
from pathlib import Path
import hashlib
import math

from gexco.config import default_config, load_config, resolve_config


def validate_architecture(config):
    """Reject unsupported YAML architecture changes before building a model."""
    defaults = load_config(default_config("reconstruction"))
    for section in ("encoder", "denoiser"):
        values = config.get(section, {})
        for key, value in values.items():
            if section == "encoder" and key in {"checkpoint", "trainable"}:
                continue
            if key not in defaults[section] or value != defaults[section][key]:
                raise ValueError(
                    f"Unsupported reconstruction architecture: {section}.{key}={value!r}; "
                    f"the paper model requires {defaults[section].get(key)!r}"
                )
    for key in ("name", "spatial_downsample", "latent_channels"):
        value = config.get("vae", {}).get(key, defaults["vae"][key])
        if value != defaults["vae"][key]:
            raise ValueError(f"Unsupported reconstruction architecture: vae.{key}={value!r}")


def validate_world_model_initialization(config):
    """Check pretraining metadata before initializing the fixed reconstruction encoder."""
    expected = {
        "in_channels": 1, "input_size": 128, "patch_size": 8, "embed_dim": 128,
        "depth": 12, "num_heads": 4, "mlp_ratio": 4.0, "use_projector": True,
        "proj_hidden_dim": 512, "proj_dim": 128, "pose_scale_learnable": True,
        "no_encoder_pose": False,
    }
    for key, value in expected.items():
        if config.get(key, value) != value:
            raise ValueError(f"World-model architecture is incompatible with reconstruction: {key} must be {value!r}")


def statistics_sha256(config):
    """Fingerprint the exact normalization statistics used for conditioning."""
    path = config.get("drr", {}).get("stats_path")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def checkpoint_configuration(payload, config_path=None, *, requested_config=None):
    """Use saved settings by default; permit sampling and file-path overrides."""
    saved = payload.get("config")
    if saved is not None and not isinstance(saved, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    saved = resolve_config(saved) if saved else None
    if requested_config is not None:
        config = resolve_config(requested_config)
    elif config_path is not None:
        config = load_config(config_path)
    else:
        config = saved or load_config(default_config("reconstruction"))
    validate_architecture(config)
    if saved is not None:
        validate_architecture(saved)
        defaults = load_config(default_config("reconstruction"))
        for section in ("data", "drr", "vae", "rectified_flow"):
            for key, expected in saved.get(section, {}).items():
                if key in {"checkpoint", "stats_path"}:
                    continue
                actual = config.get(section, {}).get(key, defaults[section].get(key))
                if actual != expected:
                    raise ValueError(f"Checkpoint configuration mismatch: {section}.{key}: {actual!r} != {expected!r}")

    expected_hash = payload.get("drr_stats_sha256")
    saved_path = saved.get("drr", {}).get("stats_path") if saved else None
    current_path = config.get("drr", {}).get("stats_path")
    if not expected_hash and saved_path and current_path != saved_path:
        if not Path(saved_path).is_file():
            raise ValueError("Cannot verify relocated normalization statistics: checkpoint has no statistics checksum")
        expected_hash = statistics_sha256(saved)
    if expected_hash and statistics_sha256(config) != expected_hash:
        raise ValueError("Checkpoint configuration mismatch: conditioning normalization statistics")
    for metadata_key, section, key in (
        ("scale_factor", "vae", "scale_factor"),
        ("num_train_timesteps", "rectified_flow", "num_train_timesteps"),
    ):
        if metadata_key in payload and not math.isclose(
            float(payload[metadata_key]), float(config[section][key]), rel_tol=1e-7
        ):
            raise ValueError(f"Checkpoint configuration mismatch: {section}.{key}")
    return config
