"""Prepare native NIfTI CTs for the fixed CT128 contract."""

import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from gexco.data.io import save_ct


def preprocess(source, destination):
    image = nib.as_closest_canonical(nib.load(str(source)))
    array = image.get_fdata(dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("Expected finite 3D HU CT")
    shape = np.maximum(1, np.round(np.asarray(array.shape) * np.asarray(image.header.get_zooms()[:3]) / 3.0)).astype(
        int
    )
    tensor = torch.from_numpy(np.clip(array, -1000.0, 1000.0))[None, None]
    # Match the project preprocessing: approximate 3-mm resampling, then fixed
    # 128-cube resize. This imposes a fixed 384-mm declared FOV and may rescale anatomy.
    tensor = F.interpolate(tensor, size=tuple(map(int, shape)), mode="trilinear", align_corners=False)
    tensor = F.interpolate(tensor, size=(128, 128, 128), mode="trilinear", align_corners=False)
    save_ct(destination, tensor[0, 0].numpy())
    return {
        "source_shape": list(array.shape),
        "intermediate_shape": shape.tolist(),
        "shape": [128, 128, 128],
        "spacing_mm": [3.0, 3.0, 3.0],
        "hu_clip": [-1000, 1000],
        "fov_note": "Fixed-cube resize can introduce patient-dependent anatomical scaling",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = preprocess(args.input, args.output)
    Path(str(args.output) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
