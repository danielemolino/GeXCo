"""Compare DiffDRR and NanoDRR on one CT using the release density contract."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from gexco.config import default_config, load_config
from gexco.data.drr_rendering import render_ct


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compare(
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    device: str | None = None,
    angles: list[float] | None = None,
    density_mode: str | None = None,
) -> dict:
    config = load_config(config_path)
    drr = config["drr"]
    if density_mode is not None:
        drr["density_mode"] = density_mode
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    angles = list(map(float, drr["angles_deg"])) if angles is None else list(map(float, angles))
    results = {}
    for renderer in ("diffdrr", "nanodrr"):
        current = copy.deepcopy(drr)
        current["renderer"] = renderer
        _sync(device)
        start = time.perf_counter()
        images = render_ct(input_path, angles, current, device=device)
        _sync(device)
        results[renderer] = {"images": images, "seconds": time.perf_counter() - start}

    reference = results["diffdrr"]["images"]
    candidate = results["nanodrr"]["images"]
    per_view = []
    for index, angle in enumerate(angles):
        a, b = reference[index], candidate[index]
        delta = b - a
        rmse = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
        dynamic_range = float(a.max() - a.min())
        per_view.append(
            {
                "angle_deg": angle,
                "mae": float(np.mean(np.abs(delta))),
                "rmse": rmse,
                "max_abs": float(np.max(np.abs(delta))),
                "correlation": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                "psnr_db": float(20.0 * np.log10(dynamic_range / rmse)) if rmse else float("inf"),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "drr_renderer_comparison.npz", **{
        name: value["images"] for name, value in results.items()
    })
    report = {
        "input": str(input_path),
        "device": str(device),
        "density_mode": drr["density_mode"],
        "stored_sampling": drr["stored_sampling"],
        "n_samples": int(drr.get("n_samples", 500)),
        "seconds": {name: value["seconds"] for name, value in results.items()},
        "speedup_nanodrr_over_diffdrr": results["diffdrr"]["seconds"] / results["nanodrr"]["seconds"],
        "views": per_view,
    }
    (output_dir / "drr_renderer_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Preprocessed CT128 NIfTI")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=default_config("preparation"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--angles", type=float, nargs="+", default=None)
    parser.add_argument("--density-mode", choices=("diffdrr", "physical_mu"), default=None)
    args = parser.parse_args()
    print(json.dumps(compare(args.input, args.config, args.output_dir, args.device, args.angles, args.density_mode), indent=2))


if __name__ == "__main__":
    main()
