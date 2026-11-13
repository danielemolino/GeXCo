"""An epoch-boundary interruption reproduces uninterrupted CPU pretraining."""

import json
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from gexco.config import load_config, default_config
import gexco.pretraining.train as trainer


def test_world_model_epoch_resume_preserves_training_state(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    records = []
    for case in range(3):
        views = []
        for view in range(13):
            path = tmp_path / f"case_{case}_view_{view}.npy"
            np.save(path, np.random.default_rng(case * 13 + view).random((32, 32), dtype=np.float32))
            views.append({"path": str(path), "angle_deg": -90 + view * 15})
        records.append({"case_id": f"case_{case}", "views": views})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))
    config = load_config(default_config("pretraining"))
    config["encoder"].update(input_size=32, patch_size=8, embed_dim=32, depth=1, num_heads=4)
    config["predictor"].update(embed_dim=32, depth=1, num_heads=4)
    config["projector"].update(hidden_dim=64, output_dim=32)
    config["sigreg"].update(projections=16, knots=5)
    config["training"].update(epochs=2, batch_size=2, num_workers=0, lr_warmup_epochs=1)
    config["data"].update(manifest=str(manifest), normalization="none", image_size=32)
    complete = trainer.train(config, tmp_path / "complete")
    original = trainer.atomic_save

    def interrupt_after_epoch_one(payload, path):
        original(payload, path)
        if Path(path).name == "last.pt" and payload["epoch"] == 1:
            raise InterruptedError("simulated interruption after durable epoch checkpoint")

    monkeypatch.setattr(trainer, "atomic_save", interrupt_after_epoch_one)
    with pytest.raises(InterruptedError):
        trainer.train(config, tmp_path / "resumed")
    monkeypatch.setattr(trainer, "atomic_save", original)
    resumed_config = copy.deepcopy(config)
    resumed_config["training"]["resume"] = str(tmp_path / "resumed/last.pt")
    resumed = trainer.train(resumed_config, tmp_path / "resumed")
    a = torch.load(complete, map_location="cpu", weights_only=False)
    b = torch.load(resumed, map_location="cpu", weights_only=False)
    for component in ("encoder", "view_predictor", "enc_projector", "pred_projector"):
        assert all(torch.equal(a[component][key], b[component][key]) for key in a[component])
    assert a["epoch"] == b["epoch"] == 2
