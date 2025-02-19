"""Abstract base class for models.

This module provides the AbstractModel class, a template for constructing machine learning
models. It initializes common attributes such as input/output dimensions and loads architecture-
specific configurations from a global Properties instance.
"""

from typing import Any

from torch import Tensor, nn

from entities.properties import Properties


class AbstractModel(nn.Module):
    """
    Abstract base class for models.

    Sets up foundational attributes for models, including input/output dimensions and
    architecture configurations as defined in a global Properties instance. Subclasses must
    override the forward() method to implement the model's computation.
    """

    def __init__(
        self, input_dim: Any, output_dim: Any, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize the AbstractModel.

        Args:
            input_dim (Any): Dimension(s) of the model input.
            output_dim (Any): Dimension(s) of the model output.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Attributes:
            properties: Global configuration instance with model settings.
            architecture: Identifier for the model architecture from properties.
            model_components: Dictionary of additional components specific to the architecture.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.properties = Properties.get_instance()
        self.architecture = self.properties.model.architecture
        self.model_components = self.properties.model.components.get(
            self.architecture, {}
        )

    def forward(self, x: Tensor) -> Any:
        """
        Compute the forward pass of the model.

        Args:
            x (Tensor): Input tensor to the model.

        Returns:
            Any: Output tensor resulting from the model's computation.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError("Forward method must be implemented in child class.")
