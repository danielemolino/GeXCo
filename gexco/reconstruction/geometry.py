"""DiffDRR-compatible projection geometry and latent world-coordinate grids.

The paper model consumes 128x128 DRRs that preserve the field of view of the
512x512, 1.2-mm renderings. Their effective detector pitch during lifting is
therefore 4.8 mm, as recorded by the training configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ViewMetadata:
    """Physical and logical metadata for one projection.

    Args:
        view_id: Integer identifier within a sample.
        angle_deg: Absolute view angle in degrees.
        rotation: DiffDRR Euler rotation in degrees.
        translation: DiffDRR translation in millimetres.
        source_to_detector_mm: Source-to-detector distance.
        source_to_patient_mm: Source-to-patient distance.
        pixel_size_mm: Effective detector pitch at the stored resolution.
        height: Stored detector height.
        width: Stored detector width.
        convention: Euler convention used by DiffDRR.
        projection_type: ``"cone"`` or ``"parallel"``.
    """

    view_id: int
    angle_deg: float
    rotation: tuple[float, float, float] | None = None
    translation: tuple[float, float, float] | None = None
    source_to_detector_mm: float = 1500.0
    source_to_patient_mm: float = 972.0
    pixel_size_mm: float = 4.8
    height: int = 128
    width: int = 128
    convention: str = "ZXY"
    projection_type: str = "cone"
    detector_axis_convention: str = "standard"

    def __post_init__(self) -> None:
        """Validate and populate canonical pose defaults."""
        if self.source_to_detector_mm <= 0 or self.source_to_patient_mm <= 0:
            raise ValueError("Source distances must be positive.")
        if self.pixel_size_mm <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("Detector size and pitch must be positive.")
        if self.projection_type not in {"cone", "parallel"}:
            raise ValueError("projection_type must be 'cone' or 'parallel'.")
        from gexco.config import normalize_detector_convention

        object.__setattr__(self, "detector_axis_convention", normalize_detector_convention(self.detector_axis_convention))
        if self.detector_axis_convention not in {"standard", "physical_bank"}:
            raise ValueError("Unknown detector axis convention")
        if self.rotation is None:
            object.__setattr__(self, "rotation", (float(self.angle_deg), 0.0, 0.0))
        if self.translation is None:
            object.__setattr__(
                self,
                "translation",
                (0.0, float(self.source_to_patient_mm), 0.0),
            )

    @property
    def sdd(self) -> float:
        """Return source-to-detector distance.

        Returns:
            Source-to-detector distance in millimetres.
        """
        return self.source_to_detector_mm

    @property
    def sdp(self) -> float:
        """Return source-to-patient distance.

        Returns:
            Source-to-patient distance in millimetres.
        """
        return self.source_to_patient_mm

    @property
    def sid(self) -> float:
        """Return source-to-patient distance using the common SID spelling.

        Returns:
            Source-to-patient distance in millimetres.
        """
        return self.source_to_patient_mm

    @property
    def delx(self) -> float:
        """Return effective detector pitch.

        Returns:
            Effective stored-image pitch in millimetres.
        """
        return self.pixel_size_mm


def standard_biplanar_views(
    height: int = 128,
    width: int = 128,
    sdd: float = 1500.0,
    sid: float = 972.0,
    pixel_size_mm: float = 4.8,
    detector_axis_convention: str = "standard",
) -> list[ViewMetadata]:
    """Construct the exact frontal and lateral DiffDRR metadata.

    Args:
        height: Stored detector height.
        width: Stored detector width.
        sdd: Source-to-detector distance in millimetres.
        sid: Source-to-patient distance in millimetres.
        pixel_size_mm: Effective stored-image pixel pitch.

    Returns:
        Frontal 0-degree and lateral 90-degree views.
    """
    common = {
        "source_to_detector_mm": float(sdd),
        "source_to_patient_mm": float(sid),
        "pixel_size_mm": float(pixel_size_mm),
        "height": int(height),
        "width": int(width),
        "translation": (0.0, float(sid), 0.0),
        "convention": "ZXY",
        "detector_axis_convention": detector_axis_convention,
    }
    return [
        ViewMetadata(
            view_id=0,
            angle_deg=0.0,
            rotation=(0.0, 0.0, 0.0),
            **common,
        ),
        ViewMetadata(
            view_id=1,
            angle_deg=90.0,
            rotation=(90.0, 0.0, 0.0),
            **common,
        ),
    ]


build_standard_two_view_metadata = standard_biplanar_views


def _ap_reorient(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the AP orientation transform used during DRR rendering."""
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Build a rotation matrix around one Cartesian axis."""
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        values = (one, zero, zero, zero, cosine, -sine, zero, sine, cosine)
    elif axis == "Y":
        values = (cosine, zero, sine, zero, one, zero, -sine, zero, cosine)
    elif axis == "Z":
        values = (cosine, -sine, zero, sine, cosine, zero, zero, zero, one)
    else:
        raise ValueError("Euler axes must be X, Y, or Z.")
    return torch.stack(values, dim=-1).reshape(angle.shape + (3, 3))


def _euler_angles_to_matrix(
    angles: torch.Tensor,
    convention: str,
) -> torch.Tensor:
    """Convert Euler angles to matrices using the DiffDRR multiplication order."""
    if angles.ndim == 0 or angles.shape[-1] != 3:
        raise ValueError("Euler angles must have shape (...,3).")
    if (
        len(convention) != 3
        or convention[1] in (convention[0], convention[2])
        or any(axis not in "XYZ" for axis in convention)
    ):
        raise ValueError(f"Invalid Euler convention: {convention}.")
    matrices = [_axis_angle_rotation(axis, angle) for axis, angle in zip(convention, torch.unbind(angles, dim=-1))]
    return matrices[0] @ matrices[1] @ matrices[2]


def _rigid_inverse(matrix: torch.Tensor) -> torch.Tensor:
    """Invert batched homogeneous rigid transforms."""
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    inverse_rotation = rotation.mT
    inverse_translation = -torch.einsum(
        "bij,bj->bi",
        inverse_rotation,
        translation,
    )
    inverse = torch.zeros_like(matrix)
    inverse[..., :3, :3] = inverse_rotation
    inverse[..., :3, 3] = inverse_translation
    inverse[..., 3, 3] = 1.0
    return inverse


def _apply_transform(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply batched homogeneous transforms to batched 3D points."""
    homogeneous = torch.nn.functional.pad(points, (0, 1), value=1.0)
    return torch.einsum("bij,bnj->bni", matrix[:, :3], homogeneous)


def _extrinsic(
    view: ViewMetadata,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the exact DiffDRR-compatible world-to-camera transform."""
    rotation = torch.tensor(
        [list(view.rotation or (0.0, 0.0, 0.0))],
        device=device,
        dtype=dtype,
    ).expand(batch_size, -1)
    translation = torch.tensor(
        [list(view.translation or (0.0, view.sdp, 0.0))],
        device=device,
        dtype=dtype,
    ).expand(batch_size, -1)
    rotation_matrix = _euler_angles_to_matrix(
        torch.deg2rad(rotation),
        view.convention,
    )
    camera_centre = torch.einsum(
        "bij,bj->bi",
        rotation_matrix,
        translation,
    )
    pose = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)
    pose[..., :3, :3] = rotation_matrix
    pose[..., :3, 3] = camera_centre
    pose[..., 3, 3] = 1.0
    reorient = _ap_reorient(device, dtype).unsqueeze(0).expand(batch_size, -1, -1)
    return _rigid_inverse(pose @ reorient)


def world_to_camera(points_world: torch.Tensor, view: ViewMetadata) -> torch.Tensor:
    """Transform world points to the DiffDRR camera frame.

    Args:
        points_world: Tensor shaped ``(B,...,3)``.
        view: Projection metadata.

    Returns:
        Camera points with the same shape.
    """
    if points_world.ndim < 2 or points_world.shape[-1] != 3:
        raise ValueError("points_world must have shape (...,3).")
    added_batch = points_world.ndim == 2
    batched = points_world.unsqueeze(0) if added_batch else points_world
    original_shape = batched.shape
    flat = batched.reshape(batched.shape[0], -1, 3)
    extrinsic = _extrinsic(view, flat.shape[0], flat.device, flat.dtype)
    camera = _apply_transform(extrinsic, flat)
    if view.detector_axis_convention == "physical_bank":
        camera = camera * camera.new_tensor([-1.0, -1.0, 1.0])
    camera = camera.reshape(original_shape)
    return camera.squeeze(0) if added_batch else camera


def project_points_to_detector(
    points_world: torch.Tensor,
    view: ViewMetadata,
    *,
    invalid_value: float = float("nan"),
) -> torch.Tensor:
    """Project world points to detector pixels using DiffDRR conventions.

    Args:
        points_world: Tensor shaped ``(B,...,3)``.
        view: Projection metadata.
        invalid_value: Fill value for points behind the source.

    Returns:
        Detector coordinates shaped ``(B,...,2)``.
    """
    camera = world_to_camera(points_world, view)
    centre_u = (view.width - 1) / 2.0
    centre_v = (view.height - 1) / 2.0
    if view.projection_type == "parallel":
        u = camera[..., 0] / view.delx + centre_u
        v = camera[..., 1] / view.delx + centre_v
        return torch.stack([u, v], dim=-1)
    depth = camera[..., 2]
    scale = view.sdd / depth.clamp_min(torch.finfo(depth.dtype).eps)
    u = scale * camera[..., 0] / view.delx + centre_u
    v = scale * camera[..., 1] / view.delx + centre_v
    fill = torch.full_like(u, invalid_value)
    u = torch.where(depth > 0, u, fill)
    v = torch.where(depth > 0, v, fill)
    return torch.stack([u, v], dim=-1)


project_points_to_detector_pixels = project_points_to_detector


def detector_pixels_to_rays(
    pixels: torch.Tensor,
    view: ViewMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project detector pixels into world-space rays.

    Args:
        pixels: Tensor shaped ``(B,...,2)``.
        view: Projection metadata.

    Returns:
        World-space origins and unit directions.
    """
    if pixels.ndim < 2 or pixels.shape[-1] != 2:
        raise ValueError("pixels must have a final dimension of size 2.")
    added_batch = pixels.ndim == 2
    batched = pixels.unsqueeze(0) if added_batch else pixels
    batch_size = batched.shape[0]
    centre_u = (view.width - 1) / 2.0
    centre_v = (view.height - 1) / 2.0
    axis_sign = -1.0 if view.detector_axis_convention == "physical_bank" else 1.0
    direction_camera = torch.stack(
        [
            axis_sign * (batched[..., 0] - centre_u) * view.delx / view.sdd,
            axis_sign * (batched[..., 1] - centre_v) * view.delx / view.sdd,
            torch.ones_like(batched[..., 0]),
        ],
        dim=-1,
    )
    transform = _rigid_inverse(_extrinsic(view, batch_size, batched.device, batched.dtype))
    flat_direction = direction_camera.reshape(batch_size, -1, 3)
    camera_origin = torch.zeros(
        batch_size,
        1,
        3,
        device=batched.device,
        dtype=batched.dtype,
    )
    world_origin = _apply_transform(transform, camera_origin)
    world_point = _apply_transform(transform, flat_direction)
    direction = torch.nn.functional.normalize(world_point - world_origin, dim=-1)
    origins = world_origin.expand_as(direction)
    origins = origins.reshape(*batched.shape[:-1], 3)
    direction = direction.reshape(*batched.shape[:-1], 3)
    if added_batch:
        return origins.squeeze(0), direction.squeeze(0)
    return origins, direction


def normalize_detector_pixels(
    pixels: torch.Tensor,
    view: ViewMetadata,
    *,
    clamp: bool = False,
) -> torch.Tensor:
    """Convert detector coordinates to ``grid_sample`` coordinates.

    Args:
        pixels: Detector coordinates with final ``(u,v)`` dimension.
        view: Detector dimensions and pitch.
        clamp: Restrict normalized values to ``[-1,1]``.

    Returns:
        Normalized sampling coordinates.
    """
    u = 2.0 * pixels[..., 0] / max(view.width - 1, 1) - 1.0
    v = 2.0 * pixels[..., 1] / max(view.height - 1, 1) - 1.0
    normalized = torch.stack([u, v], dim=-1)
    return normalized.clamp(-1.0, 1.0) if clamp else normalized


def build_latent_grid(
    spatial_shape: Sequence[int],
    batch_size: int = 1,
    *,
    spacing_zyx_mm: torch.Tensor | None = None,
    downsample_factor: int = 4,
    affine_matrix: torch.Tensor | None = None,
    extent_mm: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the latent voxel grid in world millimetres.

    Args:
        spatial_shape: ``(D,H,W)`` latent shape.
        batch_size: Batch size.
        spacing_zyx_mm: Source CT voxel spacing.
        downsample_factor: MAISI VAE spatial reduction.
        affine_matrix: Optional per-volume voxel-to-world affine.
        extent_mm: Convenience physical extent when spacing is unavailable.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Grid shaped ``(B,D,H,W,3)``.
    """
    if len(spatial_shape) != 3:
        raise ValueError("spatial_shape must contain D, H, and W.")
    depth, height, width = (int(value) for value in spatial_shape)
    target_device = torch.device(device or "cpu")
    if affine_matrix is not None:
        affine = affine_matrix.to(device=target_device, dtype=dtype)
        if affine.ndim == 2:
            affine = affine.unsqueeze(0)
        if affine.shape != (batch_size, 4, 4):
            raise ValueError("affine_matrix must have shape (B,4,4).")
        d = torch.arange(depth, device=target_device, dtype=dtype) + 0.5
        h = torch.arange(height, device=target_device, dtype=dtype) + 0.5
        w = torch.arange(width, device=target_device, dtype=dtype) + 0.5
        dd, hh, ww = torch.meshgrid(d, h, w, indexing="ij")
        ijk1 = torch.stack([dd, hh, ww, torch.ones_like(dd)], dim=-1)
        return torch.stack(
            [torch.matmul(ijk1, matrix.transpose(0, 1))[..., :3] for matrix in affine],
            dim=0,
        )
    if spacing_zyx_mm is None:
        physical_extent = tuple(float(value) for value in (extent_mm or (128.0, 128.0, 128.0)))
        spacing_zyx_mm = torch.tensor(
            [
                physical_extent[0] / depth / downsample_factor,
                physical_extent[1] / height / downsample_factor,
                physical_extent[2] / width / downsample_factor,
            ],
            device=target_device,
            dtype=dtype,
        )
    spacing = spacing_zyx_mm.to(device=target_device, dtype=dtype)
    if spacing.ndim == 1:
        spacing = spacing.unsqueeze(0)
    if spacing.shape[0] == 1 and batch_size > 1:
        spacing = spacing.expand(batch_size, -1)
    if spacing.shape != (batch_size, 3):
        raise ValueError("spacing_zyx_mm must have shape (3,) or (B,3).")
    latent_spacing = spacing * float(downsample_factor)
    grids: list[torch.Tensor] = []
    for sample_spacing in latent_spacing:
        sz, sy, sx = sample_spacing
        z = (torch.arange(depth, device=target_device, dtype=dtype) - (depth - 1) / 2) * sz
        y = (torch.arange(height, device=target_device, dtype=dtype) - (height - 1) / 2) * sy
        x = (torch.arange(width, device=target_device, dtype=dtype) - (width - 1) / 2) * sx
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        grids.append(torch.stack([xx, yy, zz], dim=-1))
    return torch.stack(grids, dim=0)


__all__ = [
    "ViewMetadata",
    "build_latent_grid",
    "build_standard_two_view_metadata",
    "detector_pixels_to_rays",
    "normalize_detector_pixels",
    "project_points_to_detector",
    "project_points_to_detector_pixels",
    "standard_biplanar_views",
    "world_to_camera",
]
