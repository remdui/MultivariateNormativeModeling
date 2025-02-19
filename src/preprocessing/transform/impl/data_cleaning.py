"""Module for data cleaning transform.

This module defines a PyTorch transform that performs basic data cleaning on an input tensor.
It can drop rows with missing values (NaNs) and remove duplicate rows while preserving the
original order.
"""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager
from entities.properties import Properties


class DataCleaningTransform(Transform):
    """
    Transform for cleaning data.

    This transform drops rows containing NaN values and removes duplicate rows (preserving the
    order of first occurrences). These operations can be enabled or disabled via the constructor.
    """

    def __init__(self, drop_na: bool = True, remove_duplicates: bool = True) -> None:
        """
        Initialize the data cleaning transform.

        Args:
            drop_na (bool): If True, drop rows that contain any NaN values.
            remove_duplicates (bool): If True, remove duplicate rows, keeping the first occurrence.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()
        self.drop_na = drop_na
        self.remove_duplicates = remove_duplicates

    def __call__(self, data: Tensor) -> Tensor:
        """
        Apply data cleaning transformations.

        This method performs two cleaning steps sequentially:
          1. Drops any rows with missing (NaN) values.
          2. Removes duplicate rows while preserving the order of the first occurrences.

        Args:
            data (Tensor): Input tensor with shape [num_rows, num_features].

        Returns:
            Tensor: Cleaned data tensor.
        """
        self.logger.info("Cleaning data")
        initial_row_count = data.size(0)

        # Drop rows with any NaN values if enabled.
        if self.drop_na:
            na_mask = ~torch.isnan(data).any(dim=1)
            data = data[na_mask]
            dropped_na_count = initial_row_count - data.size(0)
            self.logger.info(f"Dropped {dropped_na_count} rows due to NaNs.")

        # Remove duplicate rows if enabled.
        if self.remove_duplicates:
            prev_data_count = data.size(0)
            _, unique_indices = torch.unique(data, dim=0, return_inverse=True)
            _, first_occurrence_indices = torch.sort(unique_indices, stable=True)
            data = data[first_occurrence_indices]
            dropped_duplicate_count = prev_data_count - data.size(0)
            self.logger.info(
                f"Dropped {dropped_duplicate_count} duplicate rows while preserving order."
            )

        return data

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
