"""HDF5 round trips retain independent case/patient identities and acquisition metadata."""

import h5py
import numpy as np
import pytest

from gexco.config import default_config, load_config
from gexco.data.export_hdf5 import export
from gexco.data.hdf5 import load_view, manifest_from_hdf5
from gexco.utils.checkpoint import split_training_records


@pytest.mark.parametrize("resolution", [128, 64])
def test_export_round_trip_retains_patient_groups_and_configured_geometry(tmp_path, monkeypatch, resolution):
    import gexco.data.export_hdf5 as exporter

    config = load_config(default_config("preparation"))
    config["drr"].update(detector_size=384, pixel_size_mm=2.0, output_size=resolution)
    records = [
        {"case_id": f"scan-{i}-{j}", "patient_id": f"subject-{i}", "volume": str(tmp_path / f"ct-{i}-{j}.nii.gz")}
        for i in range(6) for j in range(2)
    ]
    monkeypatch.setattr(exporter, "render_ct", lambda volume, angles, drr: np.ones((len(angles), resolution, resolution), np.float32))
    export(records, tmp_path / "bank", config)
    shard = tmp_path / "bank/shard_00000.h5"
    with h5py.File(shard, "r") as h5:
        assert h5["index/case_id"].asstr()[:].tolist() == [r["case_id"] for r in records]
        assert h5["index/patient_id"].asstr()[:].tolist() == [r["patient_id"] for r in records]
        assert h5.attrs["effective_pixel_size_mm"] == pytest.approx(768.0 / resolution)
        assert h5.attrs["delx_128"] == 6.0
    for path in (tmp_path / "bank").glob("*.json"):
        path.unlink()
    recovered = manifest_from_hdf5(shard, biplanar=True)
    assert [(r["case_id"], r["patient_id"], r["volume"]) for r in recovered] == [
        (r["case_id"], r["patient_id"], r["volume"]) for r in records
    ]
    assert load_view(recovered[0]["views"][0]["path"]).shape == (resolution, resolution)
    train, validation = split_training_records(recovered)
    assert train and validation
    assert not {r["patient_id"] for r in train} & {r["patient_id"] for r in validation}


def test_older_bank_case_ids_are_not_treated_as_patient_ids(tmp_path):
    path = tmp_path / "bank.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs.update(density_mode="diffdrr", angles_deg=[0.0, 90.0])
        h5.create_dataset("index/patient_id", data=["train_10_a_1"], dtype=h5py.string_dtype())
        h5.create_dataset("views/128", data=np.zeros((1, 2, 128, 128), np.float32))
    record = manifest_from_hdf5(path)[0]
    assert record["case_id"] == "train_10_a_1"
    assert "patient_id" not in record
