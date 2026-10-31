"""No-download pipeline smoke using a procedural fixture or an authorized local CT."""

import argparse
from pathlib import Path
from gexco.utils.json import json_dumps


def create_test_ct(destination):
    """Write a deterministic numerical CT128 fixture without patient data."""
    import numpy as np
    from gexco.data.io import save_ct

    axis = np.linspace(-1, 1, 128, dtype=np.float32)
    x, y, z = axis[:, None, None], axis[None, :, None], axis[None, None, :]
    volume = np.full((128, 128, 128), -1000.0, np.float32)
    body = (x / 0.8) ** 2 + (y / 0.65) ** 2 + (z / 0.9) ** 2 < 1
    volume[body] = np.broadcast_to(40 + 20 * z, volume.shape)[body]
    for centre in (-0.3, 0.3):
        region = ((x - centre) / 0.23) ** 2 + (y / 0.4) ** 2 + (z / 0.65) ** 2 < 1
        volume[region] = np.broadcast_to(-700 + 100 * z, volume.shape)[region]
    core = (x / 0.13) ** 2 + ((y - 0.15) / 0.14) ** 2 + (z / 0.7) ** 2 < 1
    volume[core] = 500.0
    save_ct(destination, volume)
    return Path(destination)


def main():
    from gexco.config import default_config, load_config
    from gexco.data.drr_rendering import render_ct
    from gexco.data.export_hdf5 import export
    from gexco.pretraining.data import DrrWorldModelDataset
    from gexco.evaluation.segmentation import segmentation_metrics, MAX_HD95_MM
    from gexco.evaluation.volume_metrics import volume_metrics
    from gexco.data.io import load_ct
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-ct", type=Path, help="Optional authorized local CT128; otherwise generate a test fixture")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    example = args.input_ct or create_test_ct(args.output_dir / "test_ct128.nii.gz")
    config = load_config(default_config("preparation"))
    images = render_ct(example, [0.0, 90.0], config["drr"], device="cpu")
    assert images.shape == (2, 128, 128) and np.isfinite(images).all()
    manifests = export([{"case_id": "example", "volume": str(example)}], args.output_dir / "hdf5", config)
    wm_config = load_config(default_config("pretraining"))
    sample = DrrWorldModelDataset(manifests["world_model_manifest.json"], stats_path=wm_config["data"]["stats_path"])[0]
    assert sample["targets"].shape == (8, 1, 128, 128)
    ct, _ = load_ct(example)
    metrics = volume_metrics(ct, ct)
    empty = np.zeros((8, 8, 8), bool)
    mask = empty.copy()
    mask[3:5, 3:5, 3:5] = True
    assert segmentation_metrics(empty, mask)["hd95"] == MAX_HD95_MM
    assert segmentation_metrics(mask, empty)["hd95"] == MAX_HD95_MM
    result = {
        "status": "passed",
        "input_kind": "user_supplied_ct" if args.input_ct else "procedural_test_fixture",
        "biplanar_shape": list(images.shape),
        "hdf5_views": 13,
        "any_anchor_targets": 8,
        "identical_ct_psnr": metrics["psnr"],
        "identical_ct_ssim": metrics["ssim"],
        "missing_structure_hd95_mm": MAX_HD95_MM,
    }
    (args.output_dir / "smoke.json").write_text(json_dumps(result, indent=2) + "\n")
    print(json_dumps(result, indent=2))


if __name__ == "__main__":
    main()
