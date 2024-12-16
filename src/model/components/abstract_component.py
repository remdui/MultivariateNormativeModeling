"""Abstract base class for encoders."""

from typing import Any

from torch import Tensor, nn
from torch.nn import Module

from entities.properties import Properties
from model.layers.activation.factory import get_activation_function
from model.layers.normalization.factory import get_normalization_layer
from optimization.regularizers.factory import get_regularizer


class AbstractComponent(nn.Module):
    """Base class for encoders."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the encoder."""
        super().__init__()
        self.properties = Properties.get_instance()
        self.__initialize_activation_function()
        self.__initialize_final_activation_function()
        self.__initialize_regularization()

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

    def __initialize_final_activation_function(self) -> None:
        """Initialize the final activation function."""

        # Retrieve the activation function type from the model components
        final_activation_function = self.properties.model.final_activation_function

        # Retrieve the parameters specific to the selected activation function from activation_params
        final_activation_function_params = (
            self.properties.model.activation_function_params.get(
                final_activation_function, {}
            )
        )

        # Initialize the activation function with the unpacked activation function parameters
        self.final_activation_function = get_activation_function(
            final_activation_function, **final_activation_function_params
        )

    def __initialize_regularization(self) -> None:
        """Initialize the regularization methods."""

        # Initialize the dropout layer if specified in the model components
        if self.properties.model.dropout:
            self.dropout = get_regularizer(
                "dropout", p=self.properties.model.dropout_rate
            )
        else:
            self.dropout = None

    def _get_normalization_layer(self, *args: Any) -> Module:
        """Initialize the normalization layer.

        Args:
            *args: Additional arguments specific to the normalization layer.
        """
        # Retrieve the normalization layer type from the model components
        normalization_layer_name = self.properties.model.normalization_layer

        # Retrieve the parameters specific to the selected normalization layer from normalization_params
        normalization_layer_params = (
            self.properties.model.normalization_layer_params.get(
                normalization_layer_name, {}
            )
        )

        # Initialize the normalization layer with the unpacked normalization layer parameters
        normalization_layer = get_normalization_layer(
            normalization_layer_name, *args, **normalization_layer_params
        )

        return normalization_layer
