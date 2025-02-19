"""Module for limiting the number of samples in a dataset.

This module defines a PyTorch transform that randomly retains up to a specified maximum
number of samples (rows) from an input tensor. All other rows are set to NaN so that they
can be easily dropped by subsequent transforms. If the input data has fewer rows than the
specified limit, it is returned unmodified.
"""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class SampleLimitTransform(Transform):
    """
    Transform that randomly retains a maximum number of samples from a tensor.

    The transform shuffles the rows (if enabled) and keeps only up to `max_samples` rows.
    The remaining rows are set to NaN to mark them for removal by subsequent processing steps.
    If the number of samples in the input is already within the limit, the data is returned unchanged.
    """

    def __init__(self, max_samples: int = 1000, shuffle: bool = True) -> None:
        """
        Initialize the SampleLimitTransform.

        Args:
            max_samples (int): Maximum number of samples (rows) to retain.
            shuffle (bool): If True, rows are randomly permuted before selection.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.max_samples = max_samples
        self.shuffle = shuffle

    def __call__(self, data: Tensor) -> Tensor:
        """
        Apply the sample limit transformation to the input tensor.

        The transform retains up to `max_samples` rows from the input tensor. If `shuffle` is True,
        a random permutation of row indices is used for selection. Rows not selected are set to NaN.
        If the number of rows is less than or equal to `max_samples`, the original data is returned.

        Args:
            data (Tensor): Input tensor with shape [num_rows, num_features].

        Returns:
            Tensor: Tensor with only up to `max_samples` rows retained; unselected rows are set to NaN.
                   Note: This transform modifies the input tensor in place.
        """
        num_rows = data.size(0)
        self.logger.info(
            f"Applying SampleLimitTransform with a maximum of {self.max_samples} samples."
        )

        # If the number of rows is within the limit, return the data unchanged.
        if num_rows <= self.max_samples:
            self.logger.info(
                "The number of samples is within the limit. Returning original data."
            )
            return data

        # Determine the row indices to keep.
        if self.shuffle:
            # Get a random permutation of indices.
            perm = torch.randperm(num_rows, device=data.device)
        else:
            # Use a sequential order of indices.
            perm = torch.arange(num_rows, device=data.device)

        selected_indices = perm[: self.max_samples]

        # Create a boolean mask: True for rows to keep, False otherwise.
        keep_mask = torch.zeros(num_rows, dtype=torch.bool, device=data.device)
        keep_mask[selected_indices] = True

        # Set rows not in the keep_mask to NaN.
        data[~keep_mask] = float("nan")

        self.logger.info(
            f"Retained {self.max_samples} rows; set {num_rows - self.max_samples} rows to NaN."
        )
        return data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """
        Internal method to apply the sample limit transformation.

        This method is called by the underlying transformation framework to apply the transform.

        Args:
            inpt (Any): Input data to be transformed.
            params (dict[str, Any]): Additional parameters (unused).

        Returns:
            Any: The transformed data.
        """
        return self(inpt)
