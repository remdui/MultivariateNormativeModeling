"""Utility functions for mathematical operations."""

from typing import Any


def round_nested(obj: dict | list | tuple | float, decimals: int = 2) -> Any:
    """
    Recursively rounds all float values in a nested structure (dict, list, tuple) to a specified number of decimals.

    Args:
        obj (Union[dict, list, tuple, float]): The input object containing float values.
        decimals (int): Number of decimal places to round to. Defaults to 2.

    Returns:
        Any: A new structure with rounded float values, preserving original structure types.

    Examples:
        >>> round_nested({"a": 3.14159, "b": [1.2345, 2.3456]}, decimals=2)
        {'a': 3.14, 'b': [1.23, 2.35]}

        >>> round_nested((1.56789, 2.6789), decimals=1)
        (1.6, 2.7)
    """
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {key: round_nested(value, decimals) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_nested(item, decimals) for item in obj]
    if isinstance(obj, tuple):  # Preserve tuple immutability
        return tuple(round_nested(item, decimals) for item in obj)
    return obj
