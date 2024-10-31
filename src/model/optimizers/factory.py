"""Factory method to get the optimizer based on config."""

from collections.abc import Iterator
from typing import Any

from torch import optim


def get_optimizer(
    optimizer_type: str, parameters: Iterator, lr: float, **kwargs: Any
) -> optim.Optimizer:
    """Factory method to get the optimizer based on config."""
    if optimizer_type == "adam":
        return optim.Adam(parameters, lr=lr, **kwargs)
    if optimizer_type == "adamw":
        return optim.AdamW(parameters, lr=lr, **kwargs)
    if optimizer_type == "sgd":
        return optim.SGD(parameters, lr=lr, **kwargs)
    if optimizer_type == "adagrad":
        return optim.Adagrad(parameters, lr=lr, **kwargs)
    if optimizer_type == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, **kwargs)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")
