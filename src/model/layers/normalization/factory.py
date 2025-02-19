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

# Type alias for normalization layer classes (callables returning a torch.nn.Module)
NormalizationLayerClass = Callable[..., Module]

# Mapping for available normalization layers (private)
_NORMALIZATION_LAYER_MAPPING: dict[str, NormalizationLayerClass] = {
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
    """
    Factory method to create a normalization layer instance based on configuration.

    Args:
        layer_type (str): The type of normalization layer (e.g., 'batchnorm1d', 'layernorm').
                          The lookup is case-insensitive.
        *args: Positional arguments for the normalization layer's constructor.
        **kwargs: Additional keyword arguments for layer initialization.

    Returns:
        Module: An instance of the specified normalization layer.

    Raises:
        ValueError: If the layer type is not supported.
    """
    layer_class = _NORMALIZATION_LAYER_MAPPING.get(layer_type.lower())
    if layer_class is None:
        raise ValueError(f"Unknown normalization layer type: {layer_type}")
    return layer_class(*args, **kwargs)
