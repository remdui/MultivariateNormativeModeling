"""Factory module for creating optimizer instances."""

from typing import Any

from torch import optim

# Type alias for optimizer classes
OptimizerClass = type[optim.Optimizer]

# Mapping for available optimizers (private)
_OPTIMIZER_MAPPING: dict[str, OptimizerClass] = {
    # PyTorch optimizers supported for cuda
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    # PyTorch optimizers
    "adadelta": optim.Adadelta,
    "adafactor": optim.Adafactor,
    "adagrad": optim.Adagrad,
    "sparse_adam": optim.SparseAdam,
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS,
    "nadam": optim.NAdam,
    "radam": optim.RAdam,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
}


def get_optimizer(optimizer_type: str, *args: Any, **kwargs: Any) -> optim.Optimizer:
    """
    Factory method to get the optimizer based on configuration.

    Args:
        optimizer_type (str): The type of optimizer (e.g., 'adam', 'adamw', 'sgd').
                              The lookup is case-insensitive.
        *args: Additional positional arguments for the optimizer's constructor.
        **kwargs: Additional keyword arguments for the optimizer's constructor.

    Returns:
        optim.Optimizer: An instance of the specified optimizer.

    Raises:
        ValueError: If the optimizer type is not supported.
    """
    optimizer_class = _OPTIMIZER_MAPPING.get(optimizer_type.lower())
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return optimizer_class(*args, **kwargs)
