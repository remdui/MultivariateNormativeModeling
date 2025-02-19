"""Module for filtering datasets based on site information.

This module defines a PyTorch transform that filters rows in an input tensor by comparing
a specified column's values (representing site information) with a target site value.
Rows that do not match the target (within a small tolerance) are replaced with NaN.
If no site filtering is specified (selected_site == -1), the original data is returned.
"""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class SiteFilterTransform(Transform):
    """
    Transform for filtering dataset rows based on site information.

    This transform examines a specified column of the input tensor (using `col_id`)
    and sets rows that do not match the selected site value (within a tolerance) to NaN.
    If `selected_site` is -1, filtering is disabled and the original data is returned.
    """

    def __init__(self, selected_site: int = -1, col_id: int = 1) -> None:
        """
        Initialize the site filter transform.

        Args:
            selected_site (int): The target site value for filtering. Use -1 to disable filtering.
            col_id (int): The index of the column containing site information.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.selected_site = selected_site
        self.col_id = col_id

    def __call__(self, data: Tensor) -> Tensor:
        """
        Apply the site filter to the input tensor.

        If `selected_site` is -1, no filtering is applied and the original tensor is returned.
        Otherwise, rows with a site value differing from the selected site (within a tolerance of 1e-5)
        are replaced with NaN.

        Args:
            data (Tensor): Input tensor where one column holds site information.

        Returns:
            Tensor: Transformed tensor with non-matching rows set to NaN.
        """
        if self.selected_site == -1:
            self.logger.info("No site filtering applied. Returning original data.")
            return data

        self.logger.info(f"Filtering data for site ID: {self.selected_site}")

        # Extract the column with site information.
        site_data = data[:, self.col_id]
        tolerance = 1e-5  # Tolerance to handle floating-point imprecision.
        mask = torch.abs(site_data - self.selected_site) < tolerance
        data[~mask] = float("nan")  # Set rows not matching the target site to NaN.

        return data

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """
        Internal transform method.

        This method is used by the underlying transformation framework to apply the transform.

        Args:
            inpt (Any): Input data to be transformed.
            params (dict[str, Any]): Additional parameters (unused).

        Returns:
            Any: The transformed data.
        """
        return self(inpt)
