"""Defines the AbstractDataConverter abstract base class."""

from abc import ABC, abstractmethod
from logging import Logger

import pandas as pd

from entities.log_manager import LogManager
from util.file_utils import save_data


class AbstractDataConverter(ABC):
    """Abstract base class for data converters."""

    def __init__(self, logger: Logger = LogManager.get_logger(__name__)) -> None:
        """Initialize the AbstractDataConverter class."""
        self.data: pd.DataFrame = pd.DataFrame()
        self.logger = logger

    def convert(self, input_file_path: str, output_file_path: str) -> None:
        """Convert data from one format to another.

        Args:
            input_file_path (str): File path to the input data.
            output_file_path (str): File path to save the converted data.
        """
        self.logger.info(f"Converting {input_file_path} to {output_file_path}")
        self._load(input_file_path)
        self._save(output_file_path)
        self.logger.info(f"Data successfully converted and saved to {output_file_path}")

    @abstractmethod
    def _load(self, input_file_path: str) -> None:
        """Load the data from a file.

        Args:
            input_file_path (str): File path to load the data.
        """

    def _save(self, output_file_path: str) -> None:
        """Save the data to a file.

        Args:
            output_file_path (str): File path to save the data.
        """
        if self.data.empty:
            raise ValueError("No data to save. Please convert data before saving.")

        save_data(self.data, output_file_path)
        # self.data.to_hdf(output_file_path, key='data', mode='w', complevel=9, complib='blosc')
