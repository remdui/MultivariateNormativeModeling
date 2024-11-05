"""Factory module for creating optimizer instances."""

from typing import Any

from torch import optim

# Mapping for available optimizers
OPTIMIZER_MAPPING: dict[str, type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "adagrad": optim.Adagrad,
    "rmsprop": optim.RMSprop,
}


def get_optimizer(optimizer_type: str, *args: Any, **kwargs: Any) -> optim.Optimizer:
    """Factory method to get the optimizer based on config.

    Args:
        optimizer_type (str): The type of optimizer (e.g., 'adam', 'adamw', 'sgd').
        *args: Additional arguments specific to the optimizer.
        **kwargs: Additional parameters specific to the optimizer.

    Returns:
        optim.Optimizer: An instance of the specified optimizer.

    Raises:
        ValueError: If the optimizer type is not supported.
    """
    optimizer_class = OPTIMIZER_MAPPING.get(optimizer_type.lower())
    if not optimizer_class:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return optimizer_class(*args, **kwargs)
