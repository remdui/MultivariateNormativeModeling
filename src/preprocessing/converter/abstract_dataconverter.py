"""Defines the AbstractDataConverter abstract base class."""

from abc import ABC, abstractmethod

import pandas as pd


class AbstractDataConverter(ABC):
    """Abstract base class for data converters."""

    @abstractmethod
    def convert(self, input_file_path: str) -> pd.DataFrame:
        """Convert data from one format to another.

        Args:
            input_file_path (str): File path to the input data.
        """
