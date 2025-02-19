"""Factory module for creating model architecture instances."""

from typing import Any

from model.models.abstract_model import AbstractModel
from model.models.impl.vae import VAE

# Type alias for model classes (subclasses of AbstractModel)
ModelClass = type[AbstractModel]

# Mapping for available models (private)
_MODEL_MAPPING: dict[str, ModelClass] = {
    "vae": VAE,
}


def get_model(model_type: str, *args: Any, **kwargs: Any) -> AbstractModel:
    """
    Factory method to create a model architecture instance based on configuration.

    Args:
        model_type (str): The type of model (e.g., 'vae'). The lookup is case-insensitive.
        *args: Positional arguments for the model's initializer.
        **kwargs: Additional keyword arguments for the model's initializer.

    Returns:
        AbstractModel: An instance of the specified model.

    Raises:
        ValueError: If the model type is not supported.
    """
    model_class = _MODEL_MAPPING.get(model_type.lower())
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_class(*args, **kwargs)
