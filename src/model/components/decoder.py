"""Abstract base class for decoders."""

from torch import nn


class BaseDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self):
        """Initialize the decoder."""
        super().__init__()

    def forward(self, z):
        """Forward pass of the decoder.

        This method should be implemented in the child class.
        """
        raise NotImplementedError("Forward method not implemented in BaseDecoder.")
