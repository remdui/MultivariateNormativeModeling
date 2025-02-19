"""Factory module for applying weight initialization methods."""

from collections.abc import Callable
from typing import Any

from torch import nn
from torch.nn import init

# Mapping for available weight initialization methods (private)
_INITIALIZATION_MAPPING: dict[str, Any] = {
    "xavier_uniform": init.xavier_uniform_,
    "xavier_normal": init.xavier_normal_,
    "he_uniform": lambda w: init.kaiming_uniform_(w, nonlinearity="relu"),
    "he_normal": lambda w: init.kaiming_normal_(w, nonlinearity="relu"),
    "glorot_uniform": init.xavier_uniform_,
    "glorot_normal": init.xavier_normal_,
    "orthogonal": init.orthogonal_,
    "uniform": init.uniform_,
    "normal": init.normal_,
}


def get_weight_initializer(initializer_name: str) -> Callable[[Any], None]:
    """
    Factory method to get the weight initializer based on configuration.

    Args:
        initializer_name (str): The type of weight initializer (e.g., 'xavier_uniform').
                                  The lookup is case-insensitive.

    Returns:
        Callable: The initialization function that can be applied to model weights.

    Raises:
        ValueError: If the initializer type is not supported.
    """
    initializer = _INITIALIZATION_MAPPING.get(initializer_name.lower())
    if initializer is None:
        raise ValueError(f"Unknown weight initializer type: {initializer_name}")
    return initializer


def initialize_weights(model: nn.Module, initializer_name: str) -> None:
    """
    Apply the selected weight initializer to all applicable model parameters.

    If the initializer_name is an empty string or "lecun", no initialization is applied,
    assuming that the default initialization is already in place.

    Args:
        model (nn.Module): The model whose weights need initialization.
        initializer_name (str): The type of weight initializer.
    """
    if initializer_name == "":
        return

    # Lecun initialization is the default method in PyTorch, so we do not override it.
    if initializer_name.lower() == "lecun":
        return

    initializer = get_weight_initializer(initializer_name)

    def init_layer(layer: nn.Module) -> None:
        # Initialize weights for layers that have a 'weight' attribute.
        if hasattr(layer, "weight") and isinstance(layer.weight, nn.Parameter):
            if layer.weight.requires_grad and layer.weight.dim() >= 2:
                initializer(layer.weight)
        # Initialize biases if present.
        if hasattr(layer, "bias") and isinstance(layer.bias, nn.Parameter):
            if layer.bias.requires_grad and layer.bias.dim() == 1:
                init.zeros_(layer.bias)

    # Apply the initializer to all layers in the model.
    model.apply(init_layer)
