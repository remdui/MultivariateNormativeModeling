"""Defines the AbstractPreprocessor abstract base class."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class AbstractPreprocessor(ABC):
    """Abstract base class for data preprocessors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the preprocessor."""

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data.

        Args:
            data (pd.DataFrame): Data to be processed.

        Returns:
            pd.DataFrame: Processed data.
        """
