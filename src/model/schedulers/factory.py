"""Factory method to get the learning rate scheduler based on config."""

from torch.optim import lr_scheduler


def get_scheduler(scheduler_type, optimizer, **kwargs):
    """Factory method to get the learning rate scheduler based on config."""
    if scheduler_type == "steplr":
        return lr_scheduler.StepLR(optimizer, **kwargs)
    if scheduler_type == "exponentiallr":
        return lr_scheduler.ExponentialLR(optimizer, **kwargs)
    if scheduler_type == "cosineannealinglr":
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
