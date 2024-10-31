"""Abstract base class for encoders."""

from typing import Any

from torch import Tensor, nn


class BaseEncoder(nn.Module):
    """Base class for encoders."""

    def __init__(self) -> None:
        """Initialize the encoder."""
        super().__init__()

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the encoder.

        This method should be implemented in the child class.
        """
        raise NotImplementedError("Forward method not implemented in BaseEncoder.")
