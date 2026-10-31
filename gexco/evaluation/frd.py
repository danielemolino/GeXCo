"""Official volume-based Fréchet Radiomic Distance (frd-score 1.0.1, v1)."""

import numpy as np


def frd_for_volumes(reference_paths, prediction_paths, cache_dir, *, num_workers=1):
    """Official frd-score 1.0.1 v1, paired sets and project body-mask contract."""
    import importlib.metadata
    from pathlib import Path
    import hashlib
    import json
    import nibabel as nib
    from scipy import ndimage
    from frd_score import compute_frd as official_compute_frd

    if importlib.metadata.version("frd-score") != "1.0.1":
        raise RuntimeError("Expected official frd-score==1.0.1")
    if len(reference_paths) != len(prediction_paths) or len(reference_paths) < 2:
        raise ValueError("FRD needs at least two paired volumes")
    identities = [
        [(str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths]
        for paths in (reference_paths, prediction_paths)
    ]
    key = hashlib.sha256(json.dumps(identities).encode()).hexdigest()[:20]
    root = Path(cache_dir) / key
    dirs = []
    masks = []
    for side, paths in zip(["reference", "prediction"], [reference_paths, prediction_paths], strict=True):
        directory = root / side
        mask_dir = root / f"{side}_masks"
        directory.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(paths):
            destination = directory / f"{i:06d}.nii.gz"
            if not destination.exists():
                destination.symlink_to(Path(path).resolve())
            mask_path = mask_dir / destination.name
            if not mask_path.exists():
                image = nib.load(str(path))
                ct = image.get_fdata(dtype=np.float32)
                mask = ndimage.binary_opening(ndimage.binary_fill_holes(np.isfinite(ct) & (ct > -900.0)), iterations=1)
                labels, count = ndimage.label(mask)
                if count:
                    sizes = np.bincount(labels.ravel())
                    sizes[0] = 0
                    keep = np.flatnonzero(sizes >= 5000)
                    if not len(keep):
                        keep = np.asarray([int(np.argmax(sizes))])
                    mask = np.isin(labels, keep)
                nib.save(nib.Nifti1Image(mask.astype(np.uint8), image.affine), str(mask_path))
        dirs.append(str(directory))
        masks.append(str(mask_dir))
    value = official_compute_frd(
        dirs, paths_masks=masks, frd_version="v1", match_sample_count=True, verbose=False, num_workers=int(num_workers)
    )
    return {
        "frd": float(value),
        "implementation": "frd-score==1.0.1",
        "frd_version": "v1",
        "num_cases": len(reference_paths),
    }
