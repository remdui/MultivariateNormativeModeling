"""Module for filtering data based on sex.

This module defines a PyTorch transform that filters rows in an input tensor based on
sex information stored in a designated column. If the target sex is set to -1, no filtering
is applied and the original data is returned.
"""

from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class SexFilterTransform(Transform):
    """
    Transform for filtering data based on sex.

    This transform examines a specific column of the input tensor (determined by `col_id`)
    and returns only those rows where the sex value approximately equals the target sex value,
    using a small tolerance to account for floating-point precision. If the target sex is -1,
    filtering is disabled and the original tensor is returned.
    """

    def __init__(self, sex: int, col_id: int = 1) -> None:
        """
        Initialize the sex filter transform.

        Args:
            sex (int): Sex value to filter by (0 or 1). Use -1 to disable filtering.
            col_id (int): Index of the column containing sex information.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.sex = sex
        self.col_id = col_id

    def __call__(self, data: Tensor) -> Tensor:
        """
        Filter the data by sex.

        Args:
            data (Tensor): Input tensor where one column holds sex information.

        Returns:
            Tensor: A tensor containing only the rows matching the specified sex.
                    If filtering is disabled (sex == -1), the original tensor is returned.
        """
        if self.sex == -1:
            self.logger.info("No sex filtering applied. Returning original data.")
            return data

        self.logger.info(f"Filtering data for sex: {self.sex}")

        # Extract the column with sex information.
        sex_data = data[:, self.col_id]
        tolerance = 1e-5  # Tolerance to handle floating-point imprecision.
        mask = torch.abs(sex_data - self.sex) < tolerance
        filtered_data = data[mask]

        return filtered_data

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
