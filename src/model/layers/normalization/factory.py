"""Factory module for creating normalization layer instances."""

from collections.abc import Callable
from typing import Any

from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
    LocalResponseNorm,
    Module,
    RMSNorm,
    SyncBatchNorm,
)

# Mapping for available normalization layers
NORMALIZATION_LAYER_MAPPING: dict[str, Callable] = {
    "batchnorm1d": BatchNorm1d,
    "batchnorm2d": BatchNorm2d,
    "batchnorm3d": BatchNorm3d,
    "groupnorm": GroupNorm,
    "syncbatchnorm": SyncBatchNorm,
    "instancenorm1d": InstanceNorm1d,
    "instancenorm2d": InstanceNorm2d,
    "instancenorm3d": InstanceNorm3d,
    "layernorm": LayerNorm,
    "localresponsenorm": LocalResponseNorm,
    "rmsnorm": RMSNorm,
}


def get_normalization_layer(layer_type: str, *args: Any, **kwargs: Any) -> Module:
    """Factory method to get the normalization layer based on configuration.

    Args:
        layer_type (str): The type of normalization layer (e.g., 'batchnorm1d', 'layernorm').
        *args: Positional arguments for the normalization layer.
        **kwargs: Additional keyword arguments for layer initialization.

    Returns:
        Module: The normalization layer instance.

    Raises:
        ValueError: If the layer type is not supported.
    """
    layer_class = NORMALIZATION_LAYER_MAPPING.get(layer_type.lower())
    if not layer_class:
        raise ValueError(f"Unknown normalization layer type: {layer_type}")

    return layer_class(*args, **kwargs)
