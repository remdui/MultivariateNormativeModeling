"""Abstract base class for decoders."""

from typing import Any

from torch import Tensor, nn


class BaseDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self) -> None:
        """Initialize the decoder."""
        super().__init__()

    def forward(self, z: Tensor) -> Any:
        """Forward pass of the decoder.

        This method should be implemented in the child class.
        """
        raise NotImplementedError("Forward method not implemented in BaseDecoder.")
