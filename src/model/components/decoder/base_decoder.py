"""Abstract base class for decoders."""

from typing import Any

from torch import Tensor

from model.components.abstract_component import AbstractComponent


class BaseDecoder(AbstractComponent):
    """
    Abstract base class for decoders.

    Subclasses must implement the forward() method to define the decoding logic.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the decoder.

        Calls the initializer of the parent AbstractComponent.
        """
        super().__init__()

    def forward(self, x: Tensor) -> Any:
        """
        Execute the forward pass of the decoder.

        Args:
            x (Tensor): Input tensor to be decoded.

        Returns:
            Any: Decoded output.

        Raises:
            NotImplementedError: Must be implemented in the subclass.
        """
        raise NotImplementedError("Forward method must be implemented in child class.")
