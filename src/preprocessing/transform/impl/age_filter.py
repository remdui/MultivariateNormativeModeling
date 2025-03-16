"""Module for filtering dataset rows based on age range.

This module defines a PyTorch transform that filters rows from an input tensor based on an age range.
Rows with age values outside the specified bounds are removed. If either bound is set to -1, filtering
is disabled and the original data is returned.
"""

import pandas as pd

from entities.log_manager import LogManager


class AgeFilterTransform:
    """
    Transform for filtering data based on age range.

    This transform selects rows from an input DataFrame where the age value (in a specified column)
    is within the inclusive range [age_lowerbound, age_upperbound]. If either bound is -1, no filtering is applied.
    """

    def __init__(
        self,
        age_lowerbound: float = 0.0,
        age_upperbound: float = 100.0,
        col_name: str = "age",
    ) -> None:
        """
        Initialize the age filter transform.

        Args:
            age_lowerbound (float): Minimum age for filtering (inclusive). Set to -1 to disable filtering.
            age_upperbound (float): Maximum age for filtering (inclusive). Set to -1 to disable filtering.
            col_name (str): Column name containing age information.
        """
        self.logger = LogManager.get_logger(__name__)
        self.age_lowerbound = age_lowerbound
        self.age_upperbound = age_upperbound
        self.col_name = col_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the input DataFrame based on the specified age range.

        Args:
            data (pd.DataFrame): Input DataFrame where one column contains age values.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only rows within the specified age range.
        """
        if self.age_lowerbound == -1 or self.age_upperbound == -1:
            self.logger.info("No age filtering applied. Returning original data.")
            return data

        self.logger.info(
            f"Filtering data for age range: {self.age_lowerbound}-{self.age_upperbound}"
        )
        return data[
            (data[self.col_name] >= self.age_lowerbound)
            & (data[self.col_name] <= self.age_upperbound)
        ]
