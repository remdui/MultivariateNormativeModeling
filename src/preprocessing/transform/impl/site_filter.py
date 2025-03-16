"""Module for filtering datasets based on site information.

This module defines a PyTorch transform that filters rows in an input tensor by comparing
a specified column's values (representing site information) with a target site value.
Rows that do not match the target (within a small tolerance) are replaced with NaN.
If no site filtering is specified (selected_site == -1), the original data is returned.
"""

import pandas as pd

from entities.log_manager import LogManager


class SiteFilterTransform:
    """
    Transform for filtering dataset rows based on site information.

    This transform examines a specified column of the DataFrame (using `col_name`)
    and removes rows that do not match the selected site value.
    If `selected_site` is -1, filtering is disabled and the original DataFrame is returned.
    """

    def __init__(self, selected_site: int = -1, col_name: str = "site") -> None:
        """
        Initialize the site filter transform.

        Args:
            selected_site (int): The target site value for filtering. Use -1 to disable filtering.
            col_name (str): The name of the column containing site information.
        """
        self.logger = LogManager.get_logger(__name__)
        self.selected_site = selected_site
        self.col_name = col_name

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the site filter to the input DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with only rows matching the selected site.
        """
        if self.selected_site == -1:
            self.logger.info("No site filtering applied. Returning original data.")
            return data

        self.logger.info(f"Filtering data for site ID: {self.selected_site}")
        return data[data[self.col_name] == self.selected_site]
