"""Abstract base class for encoders."""

from typing import Any

from torch import Tensor, nn

from entities.properties import Properties
from model.layers.activation.factory import get_activation_function


class AbstractComponent(nn.Module):
    """Base class for encoders."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the encoder."""
        super().__init__()
        self.properties = Properties.get_instance()
        self.__initialize_activation_function()

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the encoder.

        This method should be implemented in the child class.
        """
        raise NotImplementedError("Forward method not implemented in BaseEncoder.")

    def __initialize_activation_function(self) -> None:
        """Initialize the activation function."""

        # Retrieve the activation function type from the model components
        activation_function = self.properties.model.activation_function

        # Retrieve the parameters specific to the selected activation function from activation_params
        activation_function_params = (
            self.properties.model.activation_function_params.get(
                activation_function, {}
            )
        )

        # Initialize the activation function with the unpacked activation function parameters
        self.activation_function = get_activation_function(
            activation_function, **activation_function_params
        )
