"""Module for limiting the number of samples in a dataset.

This module defines a PyTorch transform that randomly retains up to a specified maximum
number of samples (rows) from an input tensor. All other rows are set to NaN so that they
can be easily dropped by subsequent transforms. If the input data has fewer rows than the
specified limit, it is returned unmodified.
"""

import pandas as pd

from entities.log_manager import LogManager


class SampleLimitTransform:
    """
    Transform that randomly retains a maximum number of samples from a DataFrame.

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
        self.logger = LogManager.get_logger(__name__)
        self.max_samples = max_samples
        self.shuffle = shuffle

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the sample limit transformation to the input DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with only up to `max_samples` rows retained.
                          Unselected rows are set to NaN.
        """
        num_rows = len(data)
        self.logger.info(
            f"Applying SampleLimitTransform with a maximum of {self.max_samples} samples."
        )

        if num_rows <= self.max_samples:
            self.logger.info(
                "The number of samples is within the limit. Returning original data."
            )
            return data

        # Shuffle rows if needed
        if self.shuffle:
            data = data.sample(frac=1, random_state=42)

        # Select only the first `max_samples` rows
        selected_data = data.iloc[: self.max_samples].copy()

        self.logger.info(
            f"Retained {self.max_samples} rows; dropped {num_rows - self.max_samples} rows."
        )
        return selected_data
