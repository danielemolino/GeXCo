"""Cone-beam DRR preparation and parallel-beam tensor helpers."""

from __future__ import annotations

import math
from unittest.mock import patch
from typing import Any

import torch
import torch.nn.functional as F
from gexco.config import normalize_density_mode


def _rotate_volume(volume: torch.Tensor, angle_deg: float) -> torch.Tensor:
    if abs(float(angle_deg)) < 1e-8:
        return volume
    angle = math.radians(float(angle_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    transform = volume.new_tensor([[cosine, 0.0, sine, 0.0], [0.0, 1.0, 0.0, 0.0], [-sine, 0.0, cosine, 0.0]])
    transform = transform.unsqueeze(0).expand(volume.shape[0], -1, -1)
    grid = F.affine_grid(transform, volume.shape, align_corners=True)
    return F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def render_drr(
    volume: torch.Tensor,
    *,
    angle_deg: float = 0.0,
    normalize: bool = False,
) -> torch.Tensor:
    """Render a parallel-beam DRR by integrating a rotated attenuation volume.

    Args:
        volume: Attenuation volume shaped ``(B,1,D,H,W)``.
        angle_deg: View yaw in degrees.
        normalize: Min-max normalize each rendered image.

    Returns:
        Projection shaped ``(B,1,H,W)``.
    """
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError("volume must have shape (B,1,D,H,W).")
    projection = _rotate_volume(volume, angle_deg).sum(dim=2)
    if normalize:
        flat = projection.flatten(1)
        minimum = flat.min(dim=1).values[:, None, None, None]
        maximum = flat.max(dim=1).values[:, None, None, None]
        projection = (projection - minimum) / (maximum - minimum).clamp_min(1e-6)
    return projection


def render_biplanar_drrs(volume: torch.Tensor, *, normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Render frontal and lateral projections.

    Args:
        volume: Attenuation volume shaped ``(B,1,D,H,W)``.
        normalize: Min-max normalize projections.

    Returns:
        Tuple containing frontal and lateral DRRs.
    """
    return (
        render_drr(volume, angle_deg=0.0, normalize=normalize),
        render_drr(volume, angle_deg=90.0, normalize=normalize),
    )


def render_drr_diffdrr(volume_path: str, **renderer_kwargs: Any) -> torch.Tensor:
    """Render a cone-beam DRR with DiffDRR.

    Args:
        volume_path: Path to a CT volume accepted by DiffDRR.
        **renderer_kwargs: Arguments forwarded to ``diffdrr.drr.DRR``.

    Returns:
        A DiffDRR projection tensor.
    """
    try:
        from diffdrr.data import read
        from diffdrr.drr import DRR
    except ImportError as error:
        raise ImportError("Install the pinned GeXCo dependencies to use DiffDRR.") from error
    rotation = renderer_kwargs.pop("rotation", torch.zeros(1, 3))
    translation = renderer_kwargs.pop("translation", torch.tensor([[0.0, 972.0, 0.0]]))
    subject = read(volume_path)
    renderer = DRR(subject, **renderer_kwargs)
    return renderer(rotation, translation, parameterization="euler_angles", convention="ZXY")


def _hu_to_mu(hu):
    from nanodrr.data.preprocess import HU_BONE, MU_BONE, MU_WATER, hu_to_mu

    values = torch.from_numpy(hu.copy())[None, None]
    return hu_to_mu(values, mu_water=MU_WATER, mu_bone=MU_BONE, hu_bone=HU_BONE)


def _render_ct_diffdrr(hu, attenuation, affine, angles_deg, config, device):
    import diffdrr.data as diffdrr_data
    from diffdrr.data import read
    from diffdrr.drr import DRR
    import torchio as tio

    if attenuation is None:
        volume = torch.from_numpy(hu.copy())[None]
        subject = read(tio.ScalarImage(tensor=volume, affine=affine), orientation=config["orientation"])
    else:
        volume = attenuation.squeeze(0).cpu()
        # DiffDRR's Siddon integrator consumes the same precomputed attenuation
        # tensor as NanoDRR; bypass its HU preprocessing for this call only.
        with patch.object(diffdrr_data, "transform_hu_to_density", side_effect=lambda data, *_args, **_kwargs: data):
            subject = read(tio.ScalarImage(tensor=volume, affine=affine), orientation=config["orientation"])
    native_size = int(config["detector_size"])
    output_size = int(config["output_size"])
    size = output_size if config["stored_sampling"] == "direct" else native_size
    pitch = float(config["pixel_size_mm"]) * native_size / size
    renderer = DRR(subject, sdd=float(config["source_to_detector_mm"]), height=size, width=size, delx=pitch).to(device)
    translation = torch.tensor([[0.0, float(config["source_to_patient_mm"]), 0.0]], device=device)
    results = []
    with torch.inference_mode():
        for angle in angles_deg:
            rotation = torch.tensor([[math.radians(float(angle)), 0.0, 0.0]], device=device)
            image = renderer(rotation, translation, parameterization="euler_angles", convention="ZXY")
            if size != output_size:
                image = F.interpolate(image, size=(output_size, output_size), mode="area")
            results.append(image[0, 0].float().cpu().numpy())
    return results


def _render_ct_nanodrr(hu, attenuation, affine, angles_deg, config, device):
    from diffdrr.data import transform_hu_to_density
    from nanodrr.camera import make_rt_inv
    from nanodrr.data.subject import Subject
    from nanodrr.drr import DRR
    import torchio as tio

    density = attenuation.squeeze(0) if attenuation is not None else transform_hu_to_density(
        torch.from_numpy(hu.copy())[None], 1.0
    )
    image = tio.ScalarImage(tensor=density, affine=affine)
    subject = Subject.from_images(image, convert_to_mu=False).to(device)
    output_size = int(config["output_size"])
    native_size = int(config["detector_size"])
    size = output_size if config["stored_sampling"] == "direct" else native_size
    pitch = float(config["pixel_size_mm"]) * native_size / size
    renderer = DRR.from_carm_intrinsics(
        sdd=float(config["source_to_detector_mm"]),
        delx=pitch,
        dely=pitch,
        x0=0.0,
        y0=0.0,
        height=size,
        width=size,
        dtype=torch.float32,
        device=device,
    ).to(device)
    translation = torch.tensor([[0.0, float(config["source_to_patient_mm"]), 0.0]], device=device)
    n_samples = int(config.get("n_samples", 500))
    backend = config.get("nanodrr_backend", "auto")
    results = []
    with torch.inference_mode():
        for angle in angles_deg:
            rotation = torch.tensor([[float(angle), 0.0, 0.0]], device=device)
            rt_inv = make_rt_inv(rotation, translation, orientation=config["orientation"], isocenter=subject.isocenter)
            image = renderer(subject, rt_inv, n_samples=n_samples, backend=backend)
            if size != output_size:
                image = F.interpolate(image, size=(output_size, output_size), mode="area")
            results.append(image[0, 0].float().cpu().numpy())
    return results


def render_ct(volume_path, angles_deg, config, device=None):
    """Render DRRs with DiffDRR or NanoDRR under the selected density contract."""
    import numpy as np
    from gexco.data.io import load_ct

    array, affine = load_ct(volume_path)  # Clip HU before density conversion.
    density_mode = normalize_density_mode(config["density_mode"])
    if density_mode not in {"diffdrr", "physical_mu"}:
        raise ValueError(f"Unsupported DRR density mode: {density_mode!r}")
    attenuation = _hu_to_mu(array) if density_mode == "physical_mu" else None
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backend = config.get("renderer", "diffdrr")
    if backend == "diffdrr":
        results = _render_ct_diffdrr(array, attenuation, affine, angles_deg, config, device)
    elif backend == "nanodrr":
        results = _render_ct_nanodrr(array, attenuation, affine, angles_deg, config, device)
    else:
        raise ValueError(f"Unsupported DRR renderer: {backend!r}")
    if any(not np.isfinite(image).all() for image in results):
        raise ValueError("Non-finite DRR")
    return np.stack(results)


def main():
    """Generate the actual cone-beam frontal/lateral TIFF inputs."""
    import argparse
    import json
    from pathlib import Path
    import tifffile
    from gexco.config import load_config, default_config
    from gexco.data.io import case_id

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Preprocessed CT128 NIfTI")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--config", type=Path, default=default_config("preparation"))
    args = parser.parse_args()
    config = load_config(args.config)
    images = render_ct(args.input, [0.0, 90.0], config["drr"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.case_id or case_id(args.input)
    for suffix, image in zip(["F", "L"], images, strict=True):
        tifffile.imwrite(args.output_dir / f"{name}_{suffix}.tiff", image.astype("float32"))
    (args.output_dir / f"{name}_geometry.json").write_text(json.dumps(config["drr"], indent=2))
    print(f"Saved {name}_F.tiff and {name}_L.tiff ({images.shape[1:]}, raw float32)")


if __name__ == "__main__":
    main()
