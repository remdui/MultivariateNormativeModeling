"""Module for data cleaning transform.

This module defines a PyTorch transform that performs basic data cleaning on an input tensor.
It can drop rows with missing values (NaNs) and remove duplicate rows while preserving the
original order.
"""

import pandas as pd

from entities.log_manager import LogManager
from entities.properties import Properties


class DataCleaningTransform:
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
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()
        self.drop_na = drop_na
        self.remove_duplicates = remove_duplicates

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data cleaning transformations.

        This method performs two cleaning steps sequentially:
          1. Drops any rows with missing (NaN) values.
          2. Removes duplicate rows while preserving the order of the first occurrences.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        self.logger.info("Cleaning data")
        initial_row_count = len(data)

        # Drop rows with any NaN values if enabled.
        if self.drop_na:
            data = data.dropna()
            dropped_na_count = initial_row_count - len(data)
            self.logger.info(f"Dropped {dropped_na_count} rows due to NaNs.")

        # Remove duplicate rows if enabled.
        if self.remove_duplicates:
            prev_data_count = len(data)
            data = data.drop_duplicates(keep="first")
            dropped_duplicate_count = prev_data_count - len(data)
            self.logger.info(
                f"Dropped {dropped_duplicate_count} duplicate rows while preserving order."
            )

        return data
