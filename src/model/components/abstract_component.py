"""Abstract base class for neural network components (e.g., encoders).

This module defines AbstractComponent, which provides a common initialization
routine for components such as encoders. It automatically sets up activation functions,
a final activation function, and regularization (e.g., dropout) based on global configuration.
"""

from typing import Any

from torch import Tensor, nn
from torch.nn import Module

from entities.properties import Properties
from model.layers.activation.factory import get_activation_function
from model.layers.normalization.factory import get_normalization_layer
from optimization.regularizers.factory import get_regularizer


class AbstractComponent(nn.Module):
    """
    Abstract base class for neural network components.

    This class handles common setup tasks:
      - Initializes an activation function and a final activation function using configuration settings.
      - Sets up regularization (e.g., dropout) if specified.
      - Provides a helper to create normalization layers.

    Subclasses must implement the forward() method.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the AbstractComponent.

        Retrieves model configuration from a global Properties instance and sets up:
          - Activation function (self.activation_function)
          - Final activation function (self.final_activation_function)
          - Regularization (self.dropout)
        """
        super().__init__()
        self.properties = Properties.get_instance()
        self.__initialize_activation_function()
        self.__initialize_final_activation_function()
        self.__initialize_regularization()

    def forward(self, x: Tensor) -> Any:
        """
        Perform the forward pass.

        This method must be implemented in the subclass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Any: The output of the forward pass.

        Raises:
            NotImplementedError: If the subclass does not implement the forward method.
        """
        raise NotImplementedError("Forward method must be implemented in subclass.")

    def __initialize_activation_function(self) -> None:
        """
        Initialize the main activation function.

        Retrieves the activation function type and its parameters from the model configuration
        and instantiates the corresponding function.
        """
        activation_function = self.properties.model.activation_function
        activation_function_params = (
            self.properties.model.activation_function_params.get(
                activation_function, {}
            )
        )
        self.activation_function = get_activation_function(
            activation_function, **activation_function_params
        )

    def __initialize_final_activation_function(self) -> None:
        """
        Initialize the final activation function.

        Retrieves the final activation function type and parameters from configuration and
        instantiates the corresponding function.
        """
        final_activation_function = self.properties.model.final_activation_function
        final_activation_function_params = (
            self.properties.model.activation_function_params.get(
                final_activation_function, {}
            )
        )
        self.final_activation_function = get_activation_function(
            final_activation_function, **final_activation_function_params
        )

    def __initialize_regularization(self) -> None:
        """
        Initialize regularization for the component.

        If dropout is specified in the model configuration, initializes a dropout layer using the
        specified dropout rate; otherwise, sets dropout to None.
        """
        if self.properties.model.dropout:
            self.dropout = get_regularizer(
                "dropout", p=self.properties.model.dropout_rate
            )
        else:
            self.dropout = None

    def _get_normalization_layer(self, *args: Any) -> Module:
        """
        Create and return a normalization layer.

        Retrieves the normalization layer type and parameters from the model configuration,
        and instantiates the layer using the factory function.

        Args:
            *args: Additional positional arguments required by the normalization layer.

        Returns:
            Module: An initialized normalization layer.
        """
        normalization_layer_name = self.properties.model.normalization_layer
        normalization_layer_params = (
            self.properties.model.normalization_layer_params.get(
                normalization_layer_name, {}
            )
        )
        normalization_layer = get_normalization_layer(
            normalization_layer_name, *args, **normalization_layer_params
        )
        return normalization_layer
