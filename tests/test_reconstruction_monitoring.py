"""Optional training-set monitoring must not change optimization or resume."""

import copy
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gexco.config import default_config, load_config
from gexco.utils.tensors import seed_everything
import gexco.reconstruction.train as trainer


class TinyDataset:
    def __init__(self, *args, **kwargs):
        self.records = [{"case_id": f"training_{index}"} for index in range(6)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        value = 0.1 + index * 0.1
        return {
            "case_id": self.records[index]["case_id"],
            "views": torch.full((2, 1, 16, 16), value),
            "volume": torch.full((1, 16, 16, 16), value),
            "latent": torch.full((4, 4, 4, 4), value),
        }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = nn.Linear(1, 1)
        self.conditioner = nn.Module()
        self.conditioner.encoder = nn.Sequential(nn.Linear(1, 1), nn.BatchNorm1d(1))
        self.conditioner.modulation = nn.Linear(1, 1)
        self.conditioner.weighted_fusion = None

    def forward(self, noisy, timesteps, images, geometry, grid, **kwargs):
        feature = self.conditioner.encoder(images.mean((1, 2, 3, 4))[:, None])
        feature = self.conditioner.modulation(feature).reshape(-1, 1, 1, 1, 1)
        return noisy * self.denoiser.weight.reshape(1, 1, 1, 1, 1) + self.denoiser.bias + feature


def monitoring_stubs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "ReconstructionDataset", TinyDataset)
    monkeypatch.setattr(trainer, "build_training_model", lambda config: TinyModel())
    calls = SimpleNamespace(vae=0, sampling=[])

    def load_vae(*args, **kwargs):
        calls.vae += 1
        # Model initialization is allowed to consume RNG, but monitoring must isolate it.
        random.random()
        np.random.random()
        torch.rand(3)
        return object()

    def reconstruct(images, model, checkpoint, config, *, seed):
        assert all(not module.training for module in model.modules())
        assert not torch.is_grad_enabled()
        calls.sampling.append(float(images.mean()))
        seed_everything(seed)
        random.random()
        np.random.random()
        torch.rand(3)
        return torch.ones(1, 4, 4, 4, 4) * (images.mean() + 0.1)

    monkeypatch.setattr(trainer, "load_maisi_autoencoder", load_vae)
    monkeypatch.setattr(trainer, "reconstruct_latent", reconstruct)
    monkeypatch.setattr(trainer, "decode_maisi_latent", lambda latent, vae: torch.nn.functional.interpolate(
        latent.mean(1, keepdim=True), size=(16, 16, 16), mode="nearest"
    ))
    return calls


def test_monitoring_does_not_change_training_or_epoch_resume(tmp_path, monkeypatch):
    calls = monitoring_stubs(monkeypatch)
    config = load_config(default_config("reconstruction"))
    assert config["training"]["validation"] is False
    config["training"].update(manifest="training.json", epochs=2, batch_size=2, num_workers=0,
                              output_dir=str(tmp_path / "disabled"))
    config["encoder"]["checkpoint"] = "world_model.pt"
    config["data"]["volume_size"] = 16
    # The default must run without a VAE checkpoint or any reconstruction calls.
    disabled = trainer.train(config)
    assert calls.vae == 0 and calls.sampling == []
    assert not (tmp_path / "disabled/validation_samples.json").exists()

    enabled = copy.deepcopy(config)
    enabled["training"].update(validation=True, output_dir=str(tmp_path / "enabled"))
    enabled["vae"]["checkpoint"] = "vae.pt"
    monitored = trainer.train(enabled)
    assert calls.vae == 1 and len(calls.sampling) == 10
    history = [json.loads(line) for line in (tmp_path / "enabled/history.jsonl").read_text().splitlines()]
    assert all(set(row["validation"]) == {"psnr", "ssim"} for row in history)
    assert all(row["validation"]["psnr"] == pytest.approx(20.0, abs=1e-5) for row in history)
    selected = json.loads((tmp_path / "enabled/validation_samples.json").read_text())
    assert selected["source"] == "training.manifest"
    assert len(set(selected["case_ids"])) == 5
    assert set(selected["case_ids"]) <= {row["case_id"] for row in TinyDataset().records}
    assert not list((tmp_path / "enabled").glob("*best*"))

    original_save = trainer.atomic_save

    def interrupt(payload, path):
        original_save(payload, path)
        if payload["epoch"] == 1:
            raise InterruptedError("durable epoch checkpoint")

    monkeypatch.setattr(trainer, "atomic_save", interrupt)
    interrupted = copy.deepcopy(enabled)
    interrupted["training"]["output_dir"] = str(tmp_path / "resumed")
    with pytest.raises(InterruptedError):
        trainer.train(interrupted)
    monkeypatch.setattr(trainer, "atomic_save", original_save)
    interrupted["training"]["resume"] = str(tmp_path / "resumed/unet_3d_last.pt")
    resumed = trainer.train(interrupted)
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in (disabled, monitored, resumed)]
    for other in states[1:]:
        assert other["epoch"] == states[0]["epoch"] == 2
        assert other["loss"] == states[0]["loss"]
        assert torch.equal(other["torch_rng"], states[0]["torch_rng"])
        assert other["python_rng"] == states[0]["python_rng"]
        np.testing.assert_array_equal(other["numpy_rng"][1], states[0]["numpy_rng"][1])
        for key, tensor in states[0]["unet_state_dict"].items():
            assert torch.equal(other["unet_state_dict"][key], tensor), key


def test_five_training_samples_are_fixed_without_consuming_rng():
    random.seed(27)
    before = random.getstate()
    indices = trainer.training_validation_indices(TinyDataset(), seed=42)
    assert indices == trainer.training_validation_indices(TinyDataset(), seed=42)
    assert len(set(indices)) == 5 and all(0 <= index < 6 for index in indices)
    assert random.getstate() == before
    with pytest.raises(ValueError, match="at least five"):
        trainer.training_validation_indices([1, 2, 3, 4], seed=42)


def test_sampling_failure_restores_rng_and_frozen_module_modes(monkeypatch):
    monitoring_stubs(monkeypatch)
    model = TinyModel().train()
    model.conditioner.encoder.eval()
    initial_modes = [module.training for module in model.modules()]
    seed_everything(17)
    expected_python, expected_numpy = random.getstate(), np.random.get_state()
    expected_torch = torch.get_rng_state().clone()
    expected_hash = os.environ.get("PYTHONHASHSEED")

    def failing_sampler(*args, **kwargs):
        seed_everything(123)
        random.random()
        np.random.random()
        torch.rand(5)
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(trainer, "reconstruct_latent", failing_sampler)
    config = load_config(default_config("reconstruction"))
    with pytest.raises(RuntimeError, match="sampling failed"):
        trainer.validate_training_samples(model, TinyDataset(), config, object(), [0, 1, 2, 3, 4])
    assert [module.training for module in model.modules()] == initial_modes
    assert random.getstate() == expected_python
    np.testing.assert_array_equal(np.random.get_state()[1], expected_numpy[1])
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert os.environ.get("PYTHONHASHSEED") == expected_hash
