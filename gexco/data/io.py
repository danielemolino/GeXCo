"""Strict CT128 I/O. NIfTI array order is retained throughout, as in MONAI."""

from pathlib import Path
import numpy as np
import nibabel as nib


def case_id(path):
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def load_ct(path):
    image = nib.load(str(path))
    array = image.get_fdata(dtype=np.float32)
    if array.shape != (128, 128, 128):
        raise ValueError(f"Expected a preprocessed CT128 volume, got {array.shape}: {path}")
    if not np.allclose(image.header.get_zooms()[:3], (3, 3, 3), atol=1e-5):
        raise ValueError(f"Expected 3 mm isotropic CT: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite CT: {path}")
    return np.clip(array, -1000, 1000), image.affine


def save_ct(path, volume, affine=None):
    if affine is None:
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        affine[:3, 3] = -190.5
    affine = np.asarray(affine, dtype=np.float64)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(np.linalg.norm(affine[:3, :3], axis=0), 3.0, atol=1e-5)
    ):
        raise ValueError("CT output affine must declare 3 mm isotropic spacing")
    array = np.asarray(volume).squeeze()
    if array.shape != (128, 128, 128) or not np.isfinite(array).all():
        raise ValueError("CT output must be finite and shaped 128x128x128")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.rint(np.clip(array, -1000, 1000)).astype(np.int16), affine), str(output))
