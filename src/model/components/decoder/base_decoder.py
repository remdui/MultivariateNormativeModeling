"""Abstract base class for decoders."""

from typing import Any

from torch import Tensor

from model.components.abstract_component import AbstractComponent


class BaseDecoder(AbstractComponent):
    """Base class for decoders."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the decoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the decoder.

        This method should be implemented in the child class.
        """
        raise NotImplementedError("Forward method not implemented in BaseDecoder.")
