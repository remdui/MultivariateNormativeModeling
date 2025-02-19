"""Module for filtering dataset rows based on age range.

This module defines a PyTorch transform that filters rows from an input tensor based on an age range.
Rows with age values outside the specified bounds are removed. If either bound is set to -1, filtering
is disabled and the original data is returned.
"""

from typing import Any

from torch import Tensor
from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager


class AgeFilterTransform(Transform):
    """
    Transform for filtering data based on age range.

    This transform selects rows from an input tensor where the age value (located in a specified column)
    is within the inclusive range [age_lowerbound, age_upperbound]. If either bound is -1, no filtering is applied.
    """

    def __init__(
        self,
        age_lowerbound: float = 0.0,
        age_upperbound: float = 100.0,
        col_id: int = 1,
    ) -> None:
        """
        Initialize the age filter transform.

        Args:
            age_lowerbound (float): Minimum age for filtering (inclusive). Set to -1 to disable filtering.
            age_upperbound (float): Maximum age for filtering (inclusive). Set to -1 to disable filtering.
            col_id (int): Column index containing age information.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.age_lowerbound = age_lowerbound
        self.age_upperbound = age_upperbound
        self.col_id = col_id

    def __call__(self, data: Tensor) -> Tensor:
        """
        Filter the input tensor based on the specified age range.

        If filtering is disabled (either bound is -1), the original tensor is returned.
        Otherwise, only rows where the age (from the specified column) lies within the inclusive
        range [age_lowerbound, age_upperbound] are retained.

        Args:
            data (Tensor): Input tensor with shape [num_rows, num_features], where one column holds age values.

        Returns:
            Tensor: Filtered tensor containing only rows within the specified age range.
        """
        if self.age_lowerbound == -1 or self.age_upperbound == -1:
            self.logger.info("No age filtering applied. Returning original data.")
            return data

        self.logger.info(
            f"Filtering data for age range: {self.age_lowerbound}-{self.age_upperbound}"
        )

        # Extract the age column and create a mask for rows within the range.
        age_data = data[:, self.col_id]
        mask = (age_data >= self.age_lowerbound) & (age_data <= self.age_upperbound)
        filtered_data = data[mask]

        return filtered_data

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
