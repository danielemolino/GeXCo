# MAISI-compatible scheduler adapted from MONAI (Apache-2.0).
"""MAISI-compatible rectified-flow scheduler used by the GeXCo checkpoint.

The trained implementation regresses the reverse velocity ``data - noise`` and
integrates from timestep 1000 (noise) to timestep 0 (data).
"""

from __future__ import annotations

import torch


def timestep_transform(
    timestep: torch.Tensor | float,
    input_img_size_numel: int,
    *,
    base_img_size_numel: int = 32 * 32 * 32,
    scale: float = 1.0,
    num_train_timesteps: int = 1000,
    spatial_dim: int = 3,
) -> torch.Tensor:
    """Apply MAISI's resolution-aware timestep transform.

    Args:
        timestep: Scalar or tensor in the MAISI time coordinate.
        input_img_size_numel: Number of voxels in the current latent grid.
        base_img_size_numel: Reference latent voxel count.
        scale: Additional resolution-warp scale.
        num_train_timesteps: Maximum MAISI time coordinate.
        spatial_dim: Number of spatial dimensions.

    Returns:
        Transformed timesteps with the input shape preserved.
    """
    value = torch.as_tensor(timestep)
    normalized = value / num_train_timesteps
    spatial_ratio = (input_img_size_numel / base_img_size_numel) ** (1.0 / spatial_dim)
    ratio = spatial_ratio * scale
    transformed = ratio * normalized / (1.0 + (ratio - 1.0) * normalized)
    return transformed * num_train_timesteps


class RectifiedFlowScheduler:
    """Continuous rectified-flow training and Euler sampling scheduler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        *,
        use_discrete_timesteps: bool = False,
        use_timestep_transform: bool = True,
        sample_method: str = "uniform",
        transform_scale: float = 1.0,
        base_img_size_numel: int = 32 * 32 * 32,
    ) -> None:
        """Initialize the checkpoint scheduler.

        Args:
            num_train_timesteps: MAISI time-coordinate range.
            use_discrete_timesteps: Round sampled times to integers.
            use_timestep_transform: Apply resolution-aware time warping.
            sample_method: Currently ``"uniform"``.
            transform_scale: Additional time-warp scale.
            base_img_size_numel: Reference latent voxel count.
        """
        if num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be positive.")
        if sample_method != "uniform":
            raise ValueError("The released GeXCo checkpoint uses uniform sampling.")
        self.num_train_timesteps = int(num_train_timesteps)
        self.use_discrete_timesteps = bool(use_discrete_timesteps)
        self.use_timestep_transform = bool(use_timestep_transform)
        self.sample_method = sample_method
        self.transform_scale = float(transform_scale)
        self.base_img_size_numel = int(base_img_size_numel)
        self.timesteps = torch.empty(0)

    @staticmethod
    def interpolate(
        data: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor | float,
    ) -> torch.Tensor:
        """Interpolate from data at normalized ``t=0`` to noise at ``t=1``.

        Args:
            data: Clean latent tensor.
            noise: Gaussian noise with the same shape.
            time: Normalized scalar time or batch of times.

        Returns:
            Interpolated latent with the same shape as ``data``.
        """
        if data.shape != noise.shape:
            raise ValueError("data and noise must have identical shapes.")
        value = torch.as_tensor(time, device=data.device, dtype=data.dtype)
        if value.ndim == 0:
            value = value.expand(data.shape[0])
        value = value.reshape(data.shape[0], *([1] * (data.ndim - 1)))
        return (1.0 - value) * data + value * noise

    @staticmethod
    def velocity_target(data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Return the reverse velocity ``data - noise`` used by training.

        Args:
            data: Clean latent tensor.
            noise: Gaussian noise with the same shape.

        Returns:
            Constant reverse-trajectory velocity.
        """
        if data.shape != noise.shape:
            raise ValueError("data and noise must have identical shapes.")
        return data - noise

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the noisy latent at MAISI-scale timesteps.

        Args:
            original_samples: Clean latent batch.
            noise: Gaussian noise batch.
            timesteps: MAISI timesteps shaped ``(B,)``.

        Returns:
            Interpolated noisy latents.
        """
        normalized = timesteps.to(dtype=original_samples.dtype) / self.num_train_timesteps
        return self.interpolate(original_samples, noise, normalized)

    def sample_timesteps(
        self,
        sample_or_batch_size: torch.Tensor | int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Sample continuous uniform MAISI timesteps.

        Args:
            sample_or_batch_size: Reference latent or integer batch size.
            device: Required when an integer batch size is supplied.

        Returns:
            Timesteps shaped ``(B,)`` in ``[0,1000)``.
        """
        if isinstance(sample_or_batch_size, torch.Tensor):
            reference = sample_or_batch_size
            batch_size = reference.shape[0]
            target_device = reference.device
            spatial_shape = reference.shape[2:]
        else:
            batch_size = int(sample_or_batch_size)
            target_device = torch.device(device or "cpu")
            spatial_shape = (32, 32, 32)
        timesteps = torch.rand(batch_size, device=target_device) * self.num_train_timesteps
        if self.use_discrete_timesteps:
            timesteps = timesteps.long()
        if self.use_timestep_transform:
            voxel_count = int(torch.tensor(spatial_shape).prod())
            timesteps = timestep_transform(
                timesteps,
                voxel_count,
                base_img_size_numel=self.base_img_size_numel,
                scale=self.transform_scale,
                num_train_timesteps=self.num_train_timesteps,
                spatial_dim=len(spatial_shape),
            ).to(target_device)
        return timesteps

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str | None = None,
        input_img_size_numel: int = 32 * 32 * 32,
    ) -> None:
        """Create the descending noise-to-data inference schedule.

        Args:
            num_inference_steps: Number of reverse Euler evaluations.
            device: Device on which to create the schedule.
            input_img_size_numel: Current latent voxel count.
        """
        if not 1 <= num_inference_steps <= self.num_train_timesteps:
            raise ValueError("num_inference_steps is outside the supported range.")
        values = torch.tensor(
            [(1.0 - index / num_inference_steps) * self.num_train_timesteps for index in range(num_inference_steps)],
            device=device,
            dtype=torch.float32,
        )
        if self.use_timestep_transform:
            values = timestep_transform(
                values,
                input_img_size_numel,
                base_img_size_numel=self.base_img_size_numel,
                scale=self.transform_scale,
                num_train_timesteps=self.num_train_timesteps,
            ).to(device)
        if self.use_discrete_timesteps:
            values = values.round().long()
        self.timesteps = values

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | int,
        sample: torch.Tensor,
        *,
        next_timestep: float | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Take one reverse Euler step using ``data - noise`` velocity.

        Args:
            model_output: Predicted reverse velocity.
            timestep: Current MAISI timestep.
            sample: Current noisy latent.
            next_timestep: Next MAISI timestep, or the uniform default.

        Returns:
            Pair containing the next sample and predicted clean latent.
        """
        if next_timestep is None:
            if not len(self.timesteps):
                raise RuntimeError("Call set_timesteps before step.")
            delta = 1.0 / len(self.timesteps)
        else:
            delta = (float(timestep) - float(next_timestep)) / self.num_train_timesteps
        next_sample = sample + model_output * delta
        predicted_data = sample + model_output * float(timestep) / self.num_train_timesteps
        return next_sample, predicted_data

    def euler_step(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        time: float,
        next_time: float,
    ) -> torch.Tensor:
        """Integrate reverse velocity between normalized times.

        Args:
            sample: Current latent.
            velocity: Predicted reverse velocity.
            time: Current normalized time.
            next_time: Next normalized time.

        Returns:
            Euler-updated latent.
        """
        return sample + velocity * float(time - next_time)

    def integration_times(
        self,
        num_steps: int,
        *,
        reverse: bool = True,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return normalized endpoint-inclusive integration times.

        Args:
            num_steps: Number of integration intervals.
            reverse: Descend from noise to data when true.
            device: Optional output device.

        Returns:
            Tensor containing ``num_steps + 1`` normalized times.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        start, end = (1.0, 0.0) if reverse else (0.0, 1.0)
        return torch.linspace(start, end, num_steps + 1, device=device)


__all__ = ["RectifiedFlowScheduler", "timestep_transform"]
