"""Module for data normalization transform.

This module defines a PyTorch transform that normalizes input tensors using either
min-max scaling or z-score standardization.
"""

from typing import Any

from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager
from util.errors import UnsupportedNormalizationMethodError


class NormalizationTransform(Transform):
    """
    Transform for normalizing data.

    This transform applies either min-max normalization or z-score standardization to an input tensor.
    """

    def __init__(self, method: str = "min-max") -> None:
        """
        Initialize the normalizer.

        Args:
            method (str): Normalization method to use ('min-max' or 'z-score').
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.method = method

    def __call__(self, data: Tensor) -> Tensor:
        """
        Normalize the data using the specified method.

        Args:
            data (Tensor): Input tensor to be normalized.

        Returns:
            Tensor: Normalized data tensor.

        Raises:
            UnsupportedNormalizationMethodError: If the normalization method is not supported.
        """
        self.logger.info(f"Normalizing data using '{self.method}' method")
        if self.method == "min-max":
            # Compute min and max along each feature dimension.
            data_min, _ = data.min(dim=0, keepdim=True)
            data_max, _ = data.max(dim=0, keepdim=True)
            range_values = data_max - data_min
            # Avoid division by zero by substituting a small value.
            range_values[range_values == 0] = 1e-8
            normalized_data = (data - data_min) / range_values

        elif self.method == "z-score":
            # Compute mean and standard deviation along each feature dimension.
            data_mean = data.mean(dim=0, keepdim=True)
            data_std = data.std(dim=0, keepdim=True)
            # Avoid division by zero by substituting a small value.
            data_std[data_std == 0] = 1e-8
            normalized_data = (data - data_mean) / data_std

        else:
            raise UnsupportedNormalizationMethodError(
                f"Unknown normalization method: {self.method}"
            )

        return normalized_data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """
        Internal method to apply the sample limit transformation.

        This method is called by the underlying transformation framework to apply the transform.

        Args:
            inpt (Any): Input data to be normalized.
            params (dict[str, Any]): Additional parameters (unused).

        Returns:
            Any: The normalized data.
        """
        return self(inpt)
