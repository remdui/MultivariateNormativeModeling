"""Preprocessor for data cleaning."""

import pandas as pd

from data.preprocessing.abstract_preprocessor import AbstractPreprocessor
from entities.log_manager import LogManager


class DataCleaningPreprocessor(AbstractPreprocessor):
    """Preprocessor for data cleaning."""

    def __init__(self, drop_na: bool = True, remove_duplicates: bool = True):
        """Initialize the data cleaner.

        Args:
            drop_na (bool): Whether to drop rows with missing values.
            remove_duplicates (bool): Whether to remove duplicate rows.
        """
        self.logger = LogManager.get_logger(__name__)
        self.drop_na = drop_na
        self.remove_duplicates = remove_duplicates

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data.

        Args:
            data (pd.DataFrame): Data to be cleaned.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        self.logger.info("Cleaning data")
        if self.drop_na:
            data = data.dropna()
            self.logger.debug("Dropped rows with missing values")
        if self.remove_duplicates:
            data = data.drop_duplicates()
            self.logger.debug("Removed duplicate rows")
        return data
