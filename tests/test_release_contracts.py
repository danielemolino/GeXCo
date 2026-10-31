"""Regression checks for release data contracts and corrected evaluation."""

import json
import hashlib
from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
import pytest
import torch
import tifffile
from gexco.config import default_config, load_config
from gexco.data.hdf5 import view_reference, load_view, manifest_from_hdf5
from gexco.pretraining.data import load_grayscale_image, DrrWorldModelDataset
from gexco.evaluation.segmentation import MAX_HD95_MM, THORAX_CLASS_IDS, label_metrics, segmentation_metrics
from gexco.evaluation.volume_metrics import volume_metrics
from gexco.evaluation.classification import binary_average_precision
from gexco.utils.checkpoint import split_training_records


def test_hdf5_uint16_uses_physical_scale_and_rejects_wrong_density(tmp_path):
    path = tmp_path / "bank.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs.update(mu_scale=0.000188, rotate_canonical_k=0, angles_deg=[0.0, 90.0])
        h5.create_dataset("views/128", data=np.full((1, 2, 8, 8), 1000, dtype=np.uint16))
        h5.create_dataset("index/patient_id", data=["case"], dtype=h5py.string_dtype())
    image = load_view(view_reference(path, 0, 0))
    assert np.allclose(image, 0.188)
    with pytest.raises(ValueError, match="differs from checkpoint"):
        manifest_from_hdf5(path)
    with h5py.File(path, "a") as h5:
        h5.attrs["rotate_canonical_k"] = 1
    with pytest.raises(ValueError, match="orientation"):
        load_view(view_reference(path, 0, 0))


def test_float16_tiff_and_hdf5_normalize_identically(tmp_path):
    array = np.arange(64, dtype=np.float16).reshape(8, 8)
    tifffile.imwrite(tmp_path / "raw.tiff", array)
    path = tmp_path / "bank.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs.update(density_mode="diffdrr", rotate_canonical_k=0)
        h5.create_dataset("views/128", data=array[None, None])
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"train_global": {"p1": 0.0, "p99": 63.0}}))
    tiff = load_grayscale_image(tmp_path / "raw.tiff", 8, stats_path=stats, normalization="dataset_percentile_fixed")
    hdf5 = load_grayscale_image(
        view_reference(path, 0, 0), 8, stats_path=stats, normalization="dataset_percentile_fixed"
    )
    assert torch.equal(tiff, hdf5)


@pytest.mark.parametrize("suffix", [".tiff", ".png"])
def test_acquired_requires_dicom_metadata(tmp_path, suffix):
    from gexco.reconstruction.inference import load_input_image
    from PIL import Image

    array = np.linspace(0, 4095, 48 * 64).astype(np.uint16).reshape(48, 64)
    path = tmp_path / ("radiograph" + suffix)
    if suffix == ".tiff":
        tifffile.imwrite(path, array)
    else:
        Image.fromarray(array).save(path)
    with pytest.raises(ValueError, match="require DICOM metadata"):
        load_input_image(str(path), 128, input_kind="acquired", view="frontal")


@pytest.mark.parametrize("prediction_only", [False, True])
def test_absent_structure_penalty_and_both_absent_exclusion(prediction_only):
    assert len(THORAX_CLASS_IDS) == 29
    empty = np.zeros((8, 8, 8), dtype=bool)
    present = empty.copy()
    present[2:5, 2:5, 2:5] = True
    prediction, reference = (present, empty) if prediction_only else (empty, present)
    result = segmentation_metrics(prediction, reference)
    assert result == {"dice": 0.0, "hd95": MAX_HD95_MM, "nsd_3mm": 0.0}
    assert MAX_HD95_MM == pytest.approx(665.1075101064489)
    assert segmentation_metrics(empty, empty) == {"dice": None, "hd95": None, "nsd_3mm": None}
    labelled = label_metrics(prediction.astype(np.int16) * 10, reference.astype(np.int16) * 10)
    assert labelled["summary"] == result
    assert labelled["per_structure"]["lung_upper_lobe_left"] == result
    assert labelled["missing_structure_hd95_mm"] == MAX_HD95_MM
    assert label_metrics(empty, empty)["summary"] == {"dice": None, "hd95": None, "nsd_3mm": None}


def test_prediction_only_structure_is_included_in_hd95_mean():
    reference = np.zeros((8, 8, 8), dtype=np.int16)
    reference[1:3, 1:3, 1:3] = 10
    prediction = reference.copy()
    prediction[5:7, 5:7, 5:7] = 11
    result = label_metrics(prediction, reference)
    assert result["per_structure"]["lung_upper_lobe_left"]["hd95"] == 0.0
    assert result["per_structure"]["lung_lower_lobe_left"]["hd95"] == MAX_HD95_MM
    assert result["summary"] == {"dice": 0.5, "hd95": MAX_HD95_MM / 2.0, "nsd_3mm": 0.5}
    assert label_metrics(reference, prediction)["summary"] == result["summary"]


def test_surface_distance_in_mm_and_nsd_inclusive_at_3mm():
    a = np.zeros((12, 12, 12), bool)
    a[3:7, 3:7, 3:7] = True
    b = np.roll(a, 1, axis=0)
    result = segmentation_metrics(a, b, spacing=(3.0, 3.0, 3.0))
    assert result["hd95"] == pytest.approx(3.0)
    assert result["nsd_3mm"] == pytest.approx(1.0)


def test_hu_clipping_precedes_metrics():
    a = np.linspace(-1000, 1000, 11**3, dtype=np.float32).reshape(11, 11, 11)
    b = a.copy()
    a[0, 0, 0] = -9000.0
    b[0, 0, 0] = -1000.0
    result = volume_metrics(a, b)
    assert result["psnr"] == float("inf") and result["ssim"] == 1.0


def test_average_precision_ties_do_not_depend_on_row_order():
    assert binary_average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert binary_average_precision([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_any_anchor_rejects_replacement_and_uses_worker_seed(tmp_path):
    views = []
    for i in range(3):
        path = tmp_path / f"{i}.npy"
        np.save(path, np.ones((8, 8), np.float32) * i)
        views.append({"path": str(path), "angle_deg": i * 15.0})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"case_id": "one", "views": views}]))
    dataset = DrrWorldModelDataset(manifest, num_targets=3, normalization="none")
    with pytest.raises(ValueError, match="without replacement"):
        dataset[0]


def test_patient_holdout_never_splits_reconstructions_of_same_patient():
    rows = [{"case_id": f"train_{p}_{r}"} for p in range(20) for r in ["a_1", "b_1"]]
    train, validation = split_training_records(rows)
    patient = lambda row: row["case_id"].split("_")[1]
    assert not {patient(r) for r in train} & {patient(r) for r in validation}
    assert len(train) + len(validation) == 40


def test_diffdrr_density_contract():
    from diffdrr.data import transform_hu_to_density

    hu = torch.tensor([-1000.0, -950.0, -900.0, -850.0, -750.0, 0.0, 350.0, 1000.0])
    density = transform_hu_to_density(hu, 1.0)
    assert torch.equal(density[:4], density[0].expand(4))
    assert density[4] < density[5] < density[6] < density[7]
    assert load_config(default_config("preparation"))["drr"]["density_mode"] == "diffdrr"


def test_preprocessed_ct_matches_default_inference_grid(tmp_path):
    from gexco.data.preprocess_ct import preprocess
    from gexco.data.io import save_ct
    from gexco.evaluation.run_suite import evaluate_volume_directories

    source = tmp_path / "native.nii.gz"
    values = np.linspace(-900, 300, 12 * 16 * 20, dtype=np.float32).reshape(12, 16, 20)
    nib.save(nib.Nifti1Image(values, np.diag([2.0, 2.5, 3.0, 1.0])), source)
    reference = tmp_path / "reference" / "case.nii.gz"
    preprocess(source, reference)
    image = nib.load(reference)
    prediction = tmp_path / "prediction" / "case.nii.gz"
    save_ct(prediction, np.asarray(image.dataobj))
    np.testing.assert_array_equal(image.affine, nib.load(prediction).affine)
    result = evaluate_volume_directories(prediction.parent, reference.parent)
    assert result["summary"]["num_cases"] == 1
    assert result["summary"]["psnr"] == float("inf")
    assert result["summary"]["ssim"] == pytest.approx(1.0)


def test_procedural_smoke_fixture_is_reproducible_ct128_with_clean_header(tmp_path):
    from scripts.smoke import create_test_ct

    path = create_test_ct(tmp_path / "first.nii.gz")
    repeated = create_test_ct(tmp_path / "second.nii.gz")
    image = nib.load(str(path))
    assert image.shape == (128, 128, 128)
    assert image.header.get_zooms()[:3] == (3.0, 3.0, 3.0)
    assert image.get_data_dtype() == np.dtype("int16")
    hu = np.asarray(image.dataobj)
    np.testing.assert_array_equal(hu, np.asarray(nib.load(repeated).dataobj))
    assert np.isfinite(hu).all() and hu.min() >= -1000 and hu.max() <= 1000
    assert image.header["descrip"].item() == b""
    assert image.header["aux_file"].item() == b""
    assert not image.header.extensions
    np.testing.assert_array_equal(image.affine[:3, 3], [-190.5, -190.5, -190.5])


def test_bundled_lidc_example_is_ct128_hu_with_clean_header():
    path = Path(__file__).parents[1] / "examples" / "ct128.nii.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "fbfaa53aa46f27222b9bdb63cbf6c5b95f71df4181848713998fec99307f9ba2"
    )
    image = nib.load(str(path))
    assert image.shape == (128, 128, 128)
    assert image.header.get_zooms()[:3] == (3.0, 3.0, 3.0)
    assert image.get_data_dtype() == np.dtype("int16")
    hu = np.asarray(image.dataobj)
    assert np.isfinite(hu).all() and hu.min() >= -1000 and hu.max() <= 1000
    assert image.header["descrip"].item() == b""
    assert image.header["aux_file"].item() == b""
    assert not image.header.extensions


def test_nonfinite_ct_is_rejected_before_clipping():
    image = np.zeros((11, 11, 11), np.float32)
    image[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="Non-finite"):
        volume_metrics(image, np.zeros_like(image))


def test_geometry_convention_is_explicit_and_keeps_checkpoint_axes():
    from gexco.reconstruction.geometry import ViewMetadata, world_to_camera

    point = torch.tensor([[[20.0, -15.0, 30.0]]])
    standard = world_to_camera(point, ViewMetadata(0, 0.0))
    physical = world_to_camera(point, ViewMetadata(0, 0.0, detector_axis_convention="physical_bank"))
    assert torch.equal(physical, standard * torch.tensor([-1.0, -1.0, 1.0]))
    assert load_config(default_config("reconstruction"))["drr"]["detector_axis_convention"] == "standard"


def test_existing_bank_and_config_names_preserve_images_and_geometry(tmp_path):
    from gexco.reconstruction.geometry import ViewMetadata, project_points_to_detector

    config_path = tmp_path / "existing.yaml"
    config_path.write_text(
        "drr:\n  density_mode: legacy_diffdrr\n  detector_axis_convention: checkpoint_legacy\n"
    )
    config = load_config(config_path)
    assert config["drr"] == {"density_mode": "diffdrr", "detector_axis_convention": "standard"}
    points = torch.tensor([[[10.0, 20.0, 30.0], [-15.0, 40.0, 5.0]]])
    before = ViewMetadata(0, 0.0, detector_axis_convention="checkpoint_legacy")
    after = ViewMetadata(0, 0.0, detector_axis_convention=config["drr"]["detector_axis_convention"])
    old_pixels = project_points_to_detector(points, before)
    new_pixels = project_points_to_detector(points, after)
    assert torch.equal(old_pixels, new_pixels)

    path = tmp_path / "existing.h5"
    raw = np.arange(64, dtype=np.float32).reshape(8, 8)
    with h5py.File(path, "w") as h5:
        h5.attrs.update(density_mode="legacy_diffdrr", angles_deg=[0.0], rotate_canonical_k=0)
        h5.create_dataset("views/128", data=raw[None, None])
        h5.create_dataset("index/patient_id", data=["case"], dtype=h5py.string_dtype())
    manifest = manifest_from_hdf5(path)
    restored = load_view(manifest[0]["views"][0]["path"], expected_density="diffdrr")
    assert np.array_equal(restored, raw)


def test_bundled_ctnet_resolves_outside_checkout_and_loads_strictly(tmp_path, monkeypatch):
    from gexco.evaluation.ctnet import CTNetModel

    monkeypatch.chdir(tmp_path)
    checkpoint = Path(load_config(default_config("evaluation"))["classification"]["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(payload) == {"model", "label_columns"}
    assert len(payload["label_columns"]) == 18
    model = CTNetModel(len(payload["label_columns"]))
    model.load_state_dict(payload["model"], strict=True)


def test_conditioning_cache_is_per_branch_bit_exact_and_seeded():
    from torch import nn
    from gexco.reconstruction.inference import reconstruct_latent

    class Conditioner(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images, views, grid):
            self.calls += 1
            return images.mean().expand(images.shape[0], 1, *grid.shape[1:4])

    class Denoiser(nn.Module):
        def predict_velocity(self, sample, time, conditioning, spacing_tensor=None):
            return sample * 0.01 + conditioning

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.conditioner = Conditioner()
            self.denoiser = Denoiser()

        def forward(self, sample, time, images, views, grid, spacing_tensor=None):
            condition = self.conditioner(images, views, grid)
            return self.denoiser.predict_velocity(sample, time, condition, spacing_tensor=spacing_tensor)

    config = load_config(default_config("reconstruction"))
    config["data"]["volume_size"] = 16
    images = torch.ones(1, 2, 1, 128, 128)
    model = Model()
    cached = reconstruct_latent(images, model, {"scale_factor": 1.0}, config, seed=0)
    assert model.conditioner.calls == 2
    config["inference"]["conditioning_cache"] = False
    uncached = reconstruct_latent(images, model, {"scale_factor": 1.0}, config, seed=0)
    assert model.conditioner.calls == 62
    assert torch.equal(cached, uncached)
    changed = reconstruct_latent(images, model, {"scale_factor": 1.0}, config, seed=1)
    assert not torch.equal(cached, changed)
