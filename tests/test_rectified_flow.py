"""Rectified-flow path and integration tests."""

import torch

from gexco.reconstruction.rectified_flow import RectifiedFlowScheduler


def test_interpolation_endpoints_and_velocity_shape(dummy_latent: torch.Tensor) -> None:
    """Linear interpolation returns data/noise at its exact endpoints."""
    scheduler = RectifiedFlowScheduler()
    noise = torch.randn_like(dummy_latent)
    assert torch.allclose(scheduler.interpolate(dummy_latent, noise, 0.0), dummy_latent)
    assert torch.allclose(scheduler.interpolate(dummy_latent, noise, 1.0), noise)
    assert scheduler.velocity_target(dummy_latent, noise).shape == dummy_latent.shape


def test_forward_reverse_consistency(dummy_latent: torch.Tensor) -> None:
    """The trained reverse velocity maps noise to data and back exactly."""
    scheduler = RectifiedFlowScheduler()
    noise = torch.randn_like(dummy_latent)
    velocity = scheduler.velocity_target(dummy_latent, noise)
    reverse = scheduler.euler_step(noise, velocity, 1.0, 0.0)
    forward = scheduler.euler_step(reverse, velocity, 0.0, 1.0)
    assert torch.allclose(reverse, dummy_latent, atol=1e-6)
    assert torch.allclose(forward, noise, atol=1e-6)


def test_training_timesteps_use_maisi_range(dummy_latent: torch.Tensor) -> None:
    """Continuous training times occupy the serialized 0-to-1000 range."""
    scheduler = RectifiedFlowScheduler()
    timesteps = scheduler.sample_timesteps(dummy_latent)
    assert timesteps.shape == (dummy_latent.shape[0],)
    assert torch.all((timesteps >= 0) & (timesteps < 1000))
