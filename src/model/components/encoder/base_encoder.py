"""Abstract base class for encoders."""

from typing import Any

from torch import Tensor

from model.components.abstract_component import AbstractComponent


class BaseEncoder(AbstractComponent):
    """
    Abstract base class for encoders.

    Subclasses must override the forward() method to implement the encoder's logic.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the encoder.

        Calls the initializer of the parent AbstractComponent.
        """
        super().__init__()

    def forward(self, x: Tensor) -> Any:
        """
        Execute the forward pass of the encoder.

        Args:
            x (Tensor): Input tensor to be encoded.

        Returns:
            Any: The encoded representation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Forward method not implemented in BaseEncoder.")
