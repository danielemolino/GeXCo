"""Warm-up cosine learning-rate and EMA schedules used for pretraining."""

from __future__ import annotations

import math

from torch.optim import Optimizer


class WarmupCosineLRScheduler:
    """Linearly warm an optimizer and then apply cosine decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        start_lr: float,
        peak_lr: float,
        final_lr: float,
        total_steps: int,
    ) -> None:
        """Initialize the schedule.

        Args:
            optimizer: Optimizer whose groups are updated.
            warmup_steps: Linear warm-up duration.
            start_lr: First learning rate.
            peak_lr: Learning rate reached after warm-up.
            final_lr: Final cosine-decayed learning rate.
            total_steps: Total optimization steps.
        """
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.start_lr = float(start_lr)
        self.peak_lr = float(peak_lr)
        self.final_lr = float(final_lr)
        self.total_steps = max(1, int(total_steps))

    def value_at(self, step: int) -> float:
        """Return the learning rate at a zero-based step.

        Args:
            step: Zero-based optimization step.

        Returns:
            Scheduled learning rate.
        """
        step = max(0, int(step))
        if self.warmup_steps and step < self.warmup_steps:
            fraction = (step + 1) / self.warmup_steps
            return self.start_lr + fraction * (self.peak_lr - self.start_lr)
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        fraction = min(max((step - self.warmup_steps + 1) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return self.final_lr + (self.peak_lr - self.final_lr) * cosine

    def step(self, step: int) -> float:
        """Apply and return the learning rate at a step.

        Args:
            step: Zero-based optimization step.

        Returns:
            Applied learning rate.
        """
        value = self.value_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value


class CosineMomentumScheduler:
    """Cosine schedule for exponential-moving-average teacher momentum."""

    def __init__(self, start_value: float, end_value: float, total_steps: int) -> None:
        """Initialize an EMA momentum schedule.

        Args:
            start_value: Initial momentum.
            end_value: Final momentum.
            total_steps: Total optimization steps.
        """
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.total_steps = max(1, int(total_steps))

    def value_at(self, step: int) -> float:
        """Return momentum at a zero-based step.

        Args:
            step: Zero-based optimization step.

        Returns:
            Scheduled momentum.
        """
        fraction = min(max((int(step) + 1) / self.total_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return self.end_value - (self.end_value - self.start_value) * cosine


class CosineWDScheduler:
    """Cosine interpolation between initial and final AdamW weight decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        initial_weight_decay: float,
        final_weight_decay: float,
        total_steps: int,
    ) -> None:
        """Initialize the weight-decay schedule.

        Args:
            optimizer: AdamW optimizer.
            initial_weight_decay: Initial decay value.
            final_weight_decay: Final decay value.
            total_steps: Number of optimization steps.
        """
        self.optimizer = optimizer
        self.initial_weight_decay = float(initial_weight_decay)
        self.final_weight_decay = float(final_weight_decay)
        self.total_steps = max(1, int(total_steps))

    def value_at(self, step: int) -> float:
        """Return weight decay at a zero-based step.

        Args:
            step: Zero-based optimization step.

        Returns:
            Scheduled weight-decay value.
        """
        fraction = min(max((int(step) + 1) / self.total_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return self.final_weight_decay + (self.initial_weight_decay - self.final_weight_decay) * cosine

    def step(self, step: int) -> float:
        """Apply weight decay to groups not marked ``WD_exclude``.

        Args:
            step: Zero-based optimization step.

        Returns:
            Applied weight-decay value.
        """
        value = self.value_at(step)
        for group in self.optimizer.param_groups:
            if not group.get("WD_exclude", False):
                group["weight_decay"] = value
        return value


__all__ = [
    "CosineMomentumScheduler",
    "CosineWDScheduler",
    "WarmupCosineLRScheduler",
]
