"""Factory module for creating learning rate scheduler instances."""

from typing import Any

from torch.optim import lr_scheduler

# Type alias for scheduler classes
SchedulerClass = type[lr_scheduler.LRScheduler]

# Mapping for available learning rate schedulers (private)
_SCHEDULER_MAPPING: dict[str, SchedulerClass] = {
    "default": lr_scheduler.LRScheduler,
    "lambda": lr_scheduler.LambdaLR,
    "multiplicative": lr_scheduler.MultiplicativeLR,
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "constant": lr_scheduler.ConstantLR,
    "linear": lr_scheduler.LinearLR,
    "exponential": lr_scheduler.ExponentialLR,
    "polynomial": lr_scheduler.PolynomialLR,
    "cosineannealing": lr_scheduler.CosineAnnealingLR,
    "plateau": lr_scheduler.ReduceLROnPlateau,
    "cyclic": lr_scheduler.CyclicLR,
    "onecycle": lr_scheduler.OneCycleLR,
    "cosineannealingwarmrestarts": lr_scheduler.CosineAnnealingWarmRestarts,
}


def get_scheduler(
    scheduler_type: str, *args: Any, **kwargs: Any
) -> lr_scheduler.LRScheduler:
    """
    Factory method to get the learning rate scheduler based on configuration.

    Args:
        scheduler_type (str): The type of scheduler (case-insensitive).
        *args: Additional positional arguments for the scheduler's constructor.
        **kwargs: Additional keyword arguments for the scheduler's constructor.

    Returns:
        lr_scheduler.LRScheduler: An instance of the specified learning rate scheduler.

    Raises:
        ValueError: If the scheduler type is not supported.
    """
    scheduler_class = _SCHEDULER_MAPPING.get(scheduler_type.lower())
    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    return scheduler_class(*args, **kwargs)
