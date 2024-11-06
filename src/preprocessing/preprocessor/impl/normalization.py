"""Preprocessor for data normalization."""

import pandas as pd

from entities.log_manager import LogManager
from preprocessing.preprocessor.abstract_preprocessor import AbstractPreprocessor


class NormalizationPreprocessor(AbstractPreprocessor):
    """Preprocessor for data normalization."""

    def __init__(self, method: str = "min-max"):
        """Initialize the normalizer.

        Args:
            method (str): Normalization method ('min-max' or 'z-score').
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.method = method

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data.

        Args:
            data (pd.DataFrame): Data to be normalized.

        Returns:
            pd.DataFrame: Normalized data.
        """
        self.logger.info(f"Normalizing data using {self.method} method")
        if self.method == "min-max":
            # Calculate min and max values for each column
            data_min = data.min()
            data_max = data.max()
            # Avoid division by zero by setting range to 1 for columns with constant values
            range_values = data_max - data_min
            range_values[range_values == 0] = (
                1  # Replace zeros in range with 1 to avoid division by zero
            )

            return (data - data_min) / range_values
        if self.method == "z-score":
            # Calculate mean and standard deviation for each column
            data_mean = data.mean()
            data_std = data.std()

            # Replace zero standard deviations with 1 to avoid division by zero
            data_std[data_std == 0] = 1

            return (data - data_mean) / data_std
        self.logger.error(f"Unknown normalization method: {self.method}")
        raise ValueError(f"Unknown normalization method: {self.method}")
