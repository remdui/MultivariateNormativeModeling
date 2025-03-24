"""Abstract base class for neural network components (e.g., encoders).

This module defines AbstractComponent, which provides common creation methods for
components such as encoders. It automatically provides methods to create activation
functions, a final activation function, regularization (e.g., dropout), and normalization
layers based on global configuration.
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

    This class provides helper methods to create new instances for:
      - The main activation function.
      - The final activation function.
      - Regularization (e.g., dropout).
      - Normalization layers.
    Subclasses must implement the forward() method.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the AbstractComponent.

        Retrieves model configuration from a global Properties instance.
        """
        super().__init__()
        self.properties = Properties.get_instance()

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

    def get_activation_function(self) -> nn.Module:
        """
        Create and return a new main activation function.

        Retrieves the activation function type and its parameters from the model configuration
        and instantiates the corresponding function.

        Returns:
            Module: A new activation function instance.
        """
        activation_function = self.properties.model.activation_function
        activation_function_params = (
            self.properties.model.activation_function_params.get(
                activation_function, {}
            )
        )
        return get_activation_function(
            activation_function, **activation_function_params
        )

    def get_final_activation_function(self) -> nn.Module:
        """
        Create and return a new final activation function.

        Retrieves the final activation function type and parameters from configuration and
        instantiates the corresponding function.

        Returns:
            Module: A new final activation function instance.
        """
        final_activation_function = self.properties.model.final_activation_function
        final_activation_function_params = (
            self.properties.model.activation_function_params.get(
                final_activation_function, {}
            )
        )
        return get_activation_function(
            final_activation_function, **final_activation_function_params
        )

    def get_regularization(self) -> Module | None:
        """
        Create and return a new regularization layer.

        If dropout is specified in the model configuration, instantiates a dropout layer using the
        specified dropout rate; otherwise, returns None.

        Returns:
            Module or None: A new dropout layer instance or None.
        """
        if self.properties.model.dropout:
            return get_regularizer("dropout", p=self.properties.model.dropout_rate)
        return None

    def get_normalization_layer(self, *args: Any) -> Module | None:
        """
        Create and return a new normalization layer.

        Retrieves the normalization layer type and parameters from the model configuration,
        and instantiates the layer using the factory function.

        Args:
            *args: Additional positional arguments required by the normalization layer.

        Returns:
            Module or None: A new normalization layer instance or None.
        """
        if self.properties.model.normalization:
            normalization_layer_name = self.properties.model.normalization_layer
            normalization_layer_params = (
                self.properties.model.normalization_layer_params.get(
                    normalization_layer_name, {}
                )
            )
            return get_normalization_layer(
                normalization_layer_name, *args, **normalization_layer_params
            )
        return None
