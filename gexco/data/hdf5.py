"""Worker-safe HDF5 views with an explicit attenuation and orientation contract.

References are h5://<file>::<row>::<view-index>::<resolution>. Files are opened
inside each read, never shared across forked DataLoader workers.
"""

from pathlib import Path
import h5py
import numpy as np
from gexco.config import normalize_density_mode

PREFIX = "h5://"


def view_reference(path, row, view, resolution=128):
    return f"{PREFIX}{Path(path).resolve()}::{row}::{view}::{resolution}"


def load_view(reference, expected_density=None):
    source, row, view, resolution = str(reference)[len(PREFIX) :].rsplit("::", 3)
    with h5py.File(source, "r") as h5:
        if int(h5.attrs.get("rotate_canonical_k", 0)) != 0:
            raise ValueError("HDF5 arrays must use unrotated stored detector orientation")
        density = h5.attrs.get("density_mode", "physical_mu" if "mu_scale" in h5.attrs else None)
        if expected_density is not None and normalize_density_mode(density) != normalize_density_mode(expected_density):
            raise ValueError(f"HDF5 density {density!r} does not match {expected_density!r}")
        if "index/complete" in h5 and not h5["index/complete"][int(row)]:
            raise ValueError("Incomplete HDF5 row")
        dataset = h5[f"views/{resolution}"]
        array = np.asarray(dataset[int(row), int(view)], dtype=np.float32)
        if dataset.dtype == np.uint16:
            if "mu_scale" not in h5.attrs:
                raise ValueError("uint16 HDF5 views require mu_scale; never divide by 65535")
            array *= np.float32(h5.attrs["mu_scale"])
    return array


def manifest_from_hdf5(root, *, biplanar=False, expected_density="diffdrr", resolution=None):
    files = [Path(root)] if Path(root).is_file() else sorted(Path(root).glob("*.h5"))
    if not files:
        raise ValueError(f"No HDF5 shards: {root}")
    records = []
    seen = set()
    for path in files:
        with h5py.File(path, "r") as h5:
            density = h5.attrs.get("density_mode", "physical_mu" if "mu_scale" in h5.attrs else None)
            if expected_density is not None and normalize_density_mode(density) != normalize_density_mode(expected_density):
                raise ValueError(f"Bank density {density!r} differs from checkpoint density {expected_density!r}")
            angles = list(map(float, h5.attrs["angles_deg"]))
            indices = [angles.index(0.0), angles.index(90.0)] if biplanar else list(range(len(angles)))
            sizes = list(h5["views"].keys())
            selected = str(resolution) if resolution is not None else ("128" if "128" in sizes else None)
            if selected is None and len(sizes) == 1:
                selected = sizes[0]
            if selected not in sizes:
                raise ValueError(f"Select a stored HDF5 resolution from {sizes}")
            separate_ids = "index/case_id" in h5
            case_key = "index/case_id" if separate_ids else "index/patient_id"
            patients = h5["index/patient_id"].asstr()[:] if separate_ids and "index/patient_id" in h5 else None
            volumes = h5["index/ct_path"].asstr()[:] if "index/ct_path" in h5 else None
            for row, name in enumerate(h5[case_key].asstr()[:]):
                if name in seen:
                    raise ValueError(f"Duplicate case ID: {name}")
                seen.add(name)
                record = {
                    "case_id": name,
                    "views": [{"path": view_reference(path, row, i, int(selected)), "angle_deg": angles[i]} for i in indices],
                }
                if patients is not None and patients[row]:
                    record["patient_id"] = patients[row]
                if volumes is not None:
                    record["volume"] = volumes[row]
                records.append(record)
    return records
