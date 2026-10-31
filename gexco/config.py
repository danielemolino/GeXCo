"""Packaged YAML defaults and reproducible configuration overrides."""

from pathlib import Path
from copy import deepcopy
import yaml


def normalize_density_mode(value):
    """Resolve supported names for the same DiffDRR attenuation mapping."""
    return {"legacy_diffdrr": "diffdrr"}.get(value, value)


def normalize_detector_convention(value):
    """Resolve equivalent stored detector-convention names."""
    return {"checkpoint_legacy": "standard"}.get(value, value)


def default_config(stage):
    return Path(__file__).parent / "configs" / stage / "default.yaml"


def load_config(path):
    return resolve_config(yaml.safe_load(Path(path).read_text()))


def resolve_config(config):
    """Resolve packaged resources in YAML or a saved checkpoint configuration."""
    config = deepcopy(config)
    # Paths supplied by users are relative to their working directory. The bundled
    # statistics and bundled evaluation checkpoint resolve inside the package.
    for section in ("data", "drr"):
        value = config.get(section, {}).get("stats_path")
        if value and (
            value == "configs/drr_train_global_stats.json"
            or Path(value).parts[-3:] == ("gexco", "configs", "drr_train_global_stats.json")
        ):
            config[section]["stats_path"] = str(Path(__file__).parent / "configs/drr_train_global_stats.json")
    if config.get("stats_path") == "configs/drr_train_global_stats.json":
        config["stats_path"] = str(Path(__file__).parent / "configs/drr_train_global_stats.json")
    if config.get("classification", {}).get("checkpoint") == "bundled:ctnet":
        config["classification"]["checkpoint"] = str(Path(__file__).parent / "weights/ctnet.pt")
    for section in ("drr", "reprojection"):
        if "density_mode" in config.get(section, {}):
            config[section]["density_mode"] = normalize_density_mode(config[section]["density_mode"])
        if "detector_axis_convention" in config.get(section, {}):
            config[section]["detector_axis_convention"] = normalize_detector_convention(
                config[section]["detector_axis_convention"]
            )
    return config


def portable_config(config):
    """Keep the bundled statistics relocatable when saving a checkpoint."""
    config = deepcopy(config)
    bundled = str(Path(__file__).parent / "configs/drr_train_global_stats.json")
    for section in ("data", "drr"):
        if config.get(section, {}).get("stats_path") == bundled:
            config[section]["stats_path"] = "configs/drr_train_global_stats.json"
    return config
