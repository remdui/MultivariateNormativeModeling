"""Factory module for creating regularizer instances."""

from typing import Any

from torch.nn import Dropout, Module

# Type alias for regularizer classes (subclasses of nn.Module)
RegularizerClass = type[Module]

# Mapping for available regularizers (private)
_REGULARIZER_MAPPING: dict[str, RegularizerClass] = {
    "dropout": Dropout,
}


def get_regularizer(regularizer_type: str, *args: Any, **kwargs: Any) -> Module:
    """
    Factory method to create a regularizer instance based on configuration.

    Args:
        regularizer_type (str): The type of regularizer (e.g., 'dropout').
        *args: Additional positional arguments for the regularizer's constructor.
        **kwargs: Additional keyword arguments for the regularizer's constructor.

    Returns:
        Module: An instance of the requested regularizer.

    Raises:
        ValueError: If the regularizer type is not supported.
    """
    regularizer_class = _REGULARIZER_MAPPING.get(regularizer_type.lower())
    if regularizer_class is None:
        raise ValueError(f"Unknown regularizer type: {regularizer_type}")
    return regularizer_class(*args, **kwargs)
