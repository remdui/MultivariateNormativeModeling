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
            return (data - data.min()) / (data.max() - data.min())
        if self.method == "z-score":
            return (data - data.mean()) / data.std()
        self.logger.error(f"Unknown normalization method: {self.method}")
        raise ValueError(f"Unknown normalization method: {self.method}")
