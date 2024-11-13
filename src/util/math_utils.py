"""Utility functions for mathematical operations."""

from typing import Any


def round_nested(obj: Any, decimals: int = 2) -> Any:
    """Recursively round all float values in a nested structure to the specified number of decimals."""
    if isinstance(obj, dict):
        return {key: round_nested(value, decimals) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_nested(item, decimals) for item in obj]
    if isinstance(obj, float):
        return round(obj, decimals)
    return obj
