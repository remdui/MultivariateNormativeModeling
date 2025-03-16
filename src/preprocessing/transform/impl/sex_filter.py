"""Module for filtering data based on sex.

This module defines a PyTorch transform that filters rows in an input tensor based on
sex information stored in a designated column. If the target sex is set to -1, no filtering
is applied and the original data is returned.
"""

import pandas as pd

from entities.log_manager import LogManager


class SexFilterTransform:
    """
    Transform for filtering data based on sex.

    This transform examines a specific column of the DataFrame (determined by `col_name`)
    and returns only those rows where the sex value matches the target sex.
    If the target sex is -1, filtering is disabled and the original DataFrame is returned.
    """

    def __init__(self, sex: int, col_name: str = "sex") -> None:
        """
        Initialize the sex filter transform.

        Args:
            sex (int): Sex value to filter by (0 or 1). Use -1 to disable filtering.
            col_name (str): Column name containing sex information.
        """
        self.logger = LogManager.get_logger(__name__)
        self.sex = sex
        self.col_name = col_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame by sex.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only rows matching the specified sex.
        """
        if self.sex == -1:
            self.logger.info("No sex filtering applied. Returning original data.")
            return data

        self.logger.info(f"Filtering data for sex: {self.sex}")
        return data[data[self.col_name] == self.sex]
