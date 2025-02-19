"""Defines the AbstractDataConverter abstract base class.

This module provides an abstract base class for data converters that transform data
from one format to another. The conversion process follows a two-step workflow:
1. Load data from an input file (_load).
2. Save the processed data to an output file (_save).

Subclasses must implement the _load method to populate the data attribute.
"""

from abc import ABC, abstractmethod
from logging import Logger

import pandas as pd

from entities.log_manager import LogManager
from util.errors import NoDataToSaveError
from util.file_utils import save_data


class AbstractDataConverter(ABC):
    """
    Abstract base class for data converters.

    Provides a template for converting data from one file format to another by loading
    data and then saving it. Subclasses must implement the _load method to load data into
    the self.data attribute.
    """

    def __init__(self, logger: Logger = LogManager.get_logger(__name__)) -> None:
        """
        Initialize the AbstractDataConverter.

        Attributes:
            data (pd.DataFrame): The DataFrame that will hold the loaded data.
            logger (Logger): Logger instance for logging conversion events.
        """
        self.data: pd.DataFrame = pd.DataFrame()
        self.logger = logger

    def convert(self, input_file_path: str, output_file_path: str) -> None:
        """
        Convert data from one format to another.

        Loads data from the input file, then saves it to the output file.
        Logs each step of the process.

        Args:
            input_file_path (str): File path of the input data.
            output_file_path (str): File path where the converted data will be saved.

        Raises:
            NoDataToSaveError: If no data has been loaded (i.e., self.data is empty).
        """
        self.logger.info(f"Converting {input_file_path} to {output_file_path}")
        self._load(input_file_path)
        self._save(output_file_path)
        self.logger.info(f"Data successfully converted and saved to {output_file_path}")

    @abstractmethod
    def _load(self, input_file_path: str) -> None:
        """
        Load data from a file.

        Subclasses must implement this method to load data into the self.data attribute.

        Args:
            input_file_path (str): File path to load the data from.
        """

    def _save(self, output_file_path: str) -> None:
        """
        Save the data to a file.

        Saves the DataFrame stored in self.data to the specified output file using the save_data
        utility function. Raises an error if self.data is empty.

        Args:
            output_file_path (str): File path where the data will be saved.

        Raises:
            NoDataToSaveError: If self.data is empty.
        """
        if self.data.empty:
            raise NoDataToSaveError(
                "No data to save. Please convert data before saving."
            )
        save_data(self.data, output_file_path)
