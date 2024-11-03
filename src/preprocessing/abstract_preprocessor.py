"""Defines the AbstractPreprocessor abstract base class."""

from abc import ABC, abstractmethod

import pandas as pd


class AbstractPreprocessor(ABC):
    """Abstract base class for data preprocessors."""

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data.

        Args:
            data (pd.DataFrame): Data to be processed.

        Returns:
            pd.DataFrame: Processed data.
        """
