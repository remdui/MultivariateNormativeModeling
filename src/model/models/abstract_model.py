"""Abstract base class for models."""

from typing import Any

from torch import Tensor, nn

from entities.properties import Properties


class AbstractModel(nn.Module):
    """Abstract base class for models."""

    def __init__(
        self, input_dim: Any, output_dim: Any, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.properties = Properties.get_instance()
        self.architecture = self.properties.model.architecture
        self.model_components = self.properties.model.components.get(
            self.architecture, {}
        )

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the encoder.

        This method should be implemented in the child class.
        """
        raise NotImplementedError("Forward method must be implemented in child class.")
