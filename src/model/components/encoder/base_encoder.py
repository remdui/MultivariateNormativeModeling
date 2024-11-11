"""Abstract base class for encoders."""

from typing import Any

from torch import Tensor

from model.components.abstract_component import AbstractComponent


class BaseEncoder(AbstractComponent):
    """Base class for encoders."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the encoder.

        This method should be implemented in the child class.
        """
        raise NotImplementedError("Forward method not implemented in BaseEncoder.")
