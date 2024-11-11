"""Factory module for creating activation function instances."""

from typing import Any

from torch.nn import (
    CELU,
    ELU,
    GELU,
    GLU,
    SELU,
    Hardshrink,
    Hardsigmoid,
    Hardswish,
    Hardtanh,
    LeakyReLU,
    LogSigmoid,
    Mish,
    Module,
    PReLU,
    ReLU,
    ReLU6,
    RReLU,
    Sigmoid,
    SiLU,
    Softplus,
    Softshrink,
    Softsign,
    Tanh,
    Tanhshrink,
    Threshold,
)

# Mapping for available activation functions
ACTIVATION_FUNCTION_MAPPING: dict[str, Any] = {
    "elu": ELU,
    "gelu": GELU,
    "glu": GLU,
    "hardshrink": Hardshrink,
    "hardsigmoid": Hardsigmoid,
    "hardswish": Hardswish,
    "hardtanh": Hardtanh,
    "leakyrelu": LeakyReLU,
    "logsigmoid": LogSigmoid,
    "prelu": PReLU,
    "relu": ReLU,
    "relu6": ReLU6,
    "rrelu": RReLU,
    "selu": SELU,
    "celu": CELU,
    "sigmoid": Sigmoid,
    "softplus": Softplus,
    "softshrink": Softshrink,
    "softsign": Softsign,
    "tanh": Tanh,
    "tanhshrink": Tanhshrink,
    "threshold": Threshold,
    "silu": SiLU,
    "mish": Mish,
}


def get_activation_function(activation_type: str, *args: Any, **kwargs: Any) -> Module:
    """Factory method to get the activation function based on config.

    Args:
        activation_type (str): The type of activation function (e.g., 'relu', 'tanh').
        *args: Positional arguments for the activation function.
        **kwargs: Additional keyword arguments for activation function initialization.

    Returns:
        Module: The activation function instance.

    Raises:
        ValueError: If the activation function type is not supported.
    """
    activation_function_class = ACTIVATION_FUNCTION_MAPPING.get(activation_type.lower())
    if not activation_function_class:
        raise ValueError(f"Unknown activation function type: {activation_type}")
    return activation_function_class(*args, **kwargs)
