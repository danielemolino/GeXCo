"""Tensor shape, reproducibility, and device helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch.

    Args:
        seed: Reproducibility seed.
        deterministic: Request deterministic PyTorch algorithms when available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def expand_time(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast batch timesteps over a reference tensor.

    Args:
        time: Scalar or batch tensor.
        reference: Tensor defining batch and spatial dimensions.

    Returns:
        Broadcastable tensor shaped ``(B,1,...)``.
    """
    if time.ndim == 0:
        time = time.expand(reference.shape[0])
    if time.shape != (reference.shape[0],):
        raise ValueError("time must be scalar or match the reference batch size.")
    return time.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


def module_device(module: torch.nn.Module) -> torch.device:
    """Return the device of a module's first parameter.

    Args:
        module: PyTorch module containing at least one parameter.

    Returns:
        Device of the first parameter.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")
