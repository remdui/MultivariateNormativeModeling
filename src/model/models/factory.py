"""Factory module for creating model architecture instances."""

from typing import Any

from model.models.abstract_model import AbstractModel
from model.models.impl.vae import VAE

# Mapping for available models
MODEL_MAPPING: dict[str, type[AbstractModel]] = {
    "vae": VAE,
}


def get_model(
    model_type: str, input_dim: Any, output_dim: Any, *args: Any, **kwargs: Any
) -> AbstractModel:
    """Factory method to get the model architecture based on config.

    Args:
        model_type (str): The type of model (e.g., 'vae').
        input_dim (Any): Dimension of the input.
        output_dim (Any): Dimension of the output.
        *args: Positional arguments for the model initializer.
        **kwargs: Additional keyword arguments for model initialization.

    Returns:
        AbstractModel: An instance of the specified model.

    Raises:
        ValueError: If the model type is not supported.
    """
    model_class = MODEL_MAPPING.get(model_type.lower())
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_class(input_dim, output_dim, *args, **kwargs)
