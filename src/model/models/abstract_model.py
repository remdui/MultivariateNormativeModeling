"""Abstract base class for models."""

from typing import Any

from torch import Tensor, nn


class AbstractModel(nn.Module):
    """Abstract base class for models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model."""
        super().__init__()

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the encoder.

        This method should be implemented in the child class.
        """
        raise NotImplementedError("Forward method must be implemented in child class.")
