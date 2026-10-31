"""Checkpoint settings must describe the architecture and inputs actually used."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gexco.config import default_config, load_config, portable_config, resolve_config
from gexco.reconstruction.configuration import checkpoint_configuration, statistics_sha256
from gexco.reconstruction.train import build_training_model


def test_inference_uses_saved_config_and_allows_sampling_and_path_overrides(tmp_path, monkeypatch):
    import gexco.reconstruction.inference as inference

    saved = load_config(default_config("reconstruction"))
    saved["drr"]["source_to_patient_mm"] = 1000.0
    saved["inference"]["ode_steps"] = 7
    payload = {"config": portable_config(saved), "drr_stats_sha256": statistics_sha256(saved)}
    original = copy.deepcopy(payload)
    monkeypatch.setattr(inference, "restore_model", lambda *a, **kw: (torch.nn.Identity(), payload))
    _, _, config = inference.load_reconstruction_checkpoint("unused.pt", torch.device("cpu"))
    assert config["drr"]["source_to_patient_mm"] == 1000.0
    assert config["inference"]["ode_steps"] == 7
    assert payload == original

    override = copy.deepcopy(saved)
    override["inference"].update(ode_steps=9, guidance_scale=2.0, seed=12)
    override["vae"]["checkpoint"] = "checkpoints/vae.pt"
    statistics = tmp_path / "statistics.json"
    statistics.write_bytes(Path(saved["drr"]["stats_path"]).read_bytes())
    override["drr"]["stats_path"] = str(statistics)
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(override))
    _, _, config = inference.load_reconstruction_checkpoint("unused.pt", torch.device("cpu"), path)
    assert config["inference"]["ode_steps"] == 9
    assert config["vae"]["checkpoint"] == "checkpoints/vae.pt"
    assert config["drr"]["stats_path"] == str(statistics)


@pytest.mark.parametrize("section,key,value", [
    ("drr", "source_to_patient_mm", 900.0),
    ("drr", "normalization", "none"),
    ("drr", "detector_axis_convention", "physical_bank"),
    ("data", "voxel_spacing_mm", [2.0, 2.0, 2.0]),
    ("rectified_flow", "use_timestep_transform", False),
])
def test_incompatible_checkpoint_input_overrides_are_rejected(section, key, value):
    saved = load_config(default_config("reconstruction"))
    requested = copy.deepcopy(saved)
    requested[section][key] = value
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        checkpoint_configuration({"config": saved}, requested_config=requested)


@pytest.mark.parametrize("section,key,value", [
    ("denoiser", "channels", [32, 64]),
    ("denoiser", "res_blocks_per_level", 1),
    ("encoder", "num_heads", 8),
    ("encoder", "use_projector", False),
])
def test_unsupported_architecture_is_rejected_before_loading_weights(section, key, value):
    config = load_config(default_config("reconstruction"))
    config["encoder"]["checkpoint"] = "does-not-exist.pt"
    config[section][key] = value
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        build_training_model(config)


def test_incompatible_world_model_is_rejected_before_state_loading(tmp_path):
    from gexco.reconstruction.denoiser import build_reconstruction_model

    path = tmp_path / "wrong_architecture.pt"
    torch.save({"config": {"num_heads": 8}}, path)
    with pytest.raises(ValueError, match="num_heads"):
        build_reconstruction_model(pretraining_checkpoint=path)


def test_statistics_relocation_and_modified_content_are_checked(tmp_path):
    config = load_config(default_config("reconstruction"))
    path = tmp_path / "original.json"
    path.write_text('{"train_global": {"p1": 0, "p99": 10}}')
    config["drr"]["stats_path"] = str(path)
    payload = {"config": config, "drr_stats_sha256": statistics_sha256(config)}
    relocated = tmp_path / "relocated.json"
    relocated.write_bytes(path.read_bytes())
    path.unlink()
    requested = copy.deepcopy(config)
    requested["drr"]["stats_path"] = str(relocated)
    assert checkpoint_configuration(payload, requested_config=requested)["drr"]["stats_path"] == str(relocated)
    relocated.write_text('{"train_global": {"p1": 0, "p99": 20}}')
    with pytest.raises(ValueError, match="normalization statistics"):
        checkpoint_configuration(payload, requested_config=requested)


def test_checkpoint_without_saved_config_uses_defaults_and_checks_metadata():
    config = checkpoint_configuration({})
    assert config == load_config(default_config("reconstruction"))
    with pytest.raises(ValueError, match="scale_factor"):
        checkpoint_configuration({"scale_factor": 0.5})


def test_bundled_statistics_are_relocatable_without_mutating_config():
    config = load_config(default_config("reconstruction"))
    portable = portable_config(config)
    assert portable["drr"]["stats_path"] == "configs/drr_train_global_stats.json"
    assert resolve_config(portable) == config
    portable["drr"]["stats_path"] = "/opt/previous/gexco/configs/drr_train_global_stats.json"
    assert resolve_config(portable) == config


def test_evaluation_cli_writes_strict_json_for_identity_psnr(tmp_path, monkeypatch):
    from gexco.evaluation.run_suite import main

    for name in ("prediction", "reference"):
        directory = tmp_path / name
        directory.mkdir()
        np.save(directory / "case.npy", np.zeros((11, 11, 11), np.float32))
    output = tmp_path / "evaluation"
    monkeypatch.setattr("sys.argv", [
        "evaluate", "--predictions", str(tmp_path / "prediction"),
        "--targets", str(tmp_path / "reference"), "--metrics", "volume", "--output-dir", str(output),
    ])
    main()

    def reject_constant(value):
        raise AssertionError(f"Non-standard JSON constant: {value}")

    result = json.loads((output / "metrics.json").read_text(), parse_constant=reject_constant)
    assert result["summary"]["volume"]["psnr"] == "Infinity"
    assert result["summary"]["volume"]["ssim"] == 1.0


def test_json_preserves_exclusions_and_encodes_numpy_nonfinite_values():
    from gexco.utils.json import json_dumps

    encoded = json_dumps({"values": np.array([np.inf, -np.inf, np.nan]), "excluded": None, "finite": np.float32(0.5)})
    assert json.loads(encoded) == {"values": ["Infinity", "-Infinity", "NaN"], "excluded": None, "finite": 0.5}
