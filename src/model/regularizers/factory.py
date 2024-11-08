"""Factory module for creating regularizer instances."""

from typing import Any

from torch import nn

# Mapping for available regularizers
REGULARIZER_MAPPING: dict[str, Any] = {
    "dropout": lambda p=0.5: nn.Dropout(p),
}


def get_regularizer(regularizer_type: str, *args: Any, **kwargs: Any) -> Any | None:
    """Factory method to get the regularizer based on config.

    Args:
        regularizer_type (str): The type of regularizer (e.g., 'l1', 'l2', 'dropout').
        *args: Additional arguments specific to the regularizer.
        **kwargs: Additional parameters specific to the regularizer.

    Returns:
        A regularizer function or nn.Module, or None if no regularizer is specified.

    Raises:
        ValueError: If the regularizer type is not supported.
    """
    regularizer = REGULARIZER_MAPPING.get(regularizer_type.lower())
    if not regularizer:
        raise ValueError(f"Unknown regularizer type: {regularizer_type}")
    return regularizer(*args, **kwargs)
