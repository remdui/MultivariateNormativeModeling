"""Data converter for CSV files.

This module defines the CSVConverter class, which extends AbstractDataConverter to load
data from CSV files. It first reads the CSV without headers and checks if the first row
consists solely of strings (indicating the presence of headers). If so, it re-reads the CSV
with headers.
"""

import pandas as pd

from entities.log_manager import LogManager
from preprocessing.converter.abstract_data_converter import AbstractDataConverter


class CSVConverter(AbstractDataConverter):
    """
    Data converter for CSV files.

    Loads a CSV file into a pandas DataFrame. If the first row contains only strings,
    it assumes that the CSV file includes headers and re-reads the file accordingly.
    """

    def __init__(self) -> None:
        """
        Initialize the CSVConverter.

        Sets up the logger for the converter.
        """
        super().__init__(LogManager.get_logger(__name__))

    def _load(self, input_file_path: str) -> None:
        """
        Load data from a CSV file.

        The converter first attempts to read the CSV file without headers.
        It then checks if the first row contains only strings, which likely indicates headers.
        If headers are detected, the CSV is re-read with headers.

        Args:
            input_file_path (str): The path to the CSV file to be loaded.

        Raises:
            OSError: If the CSV file cannot be read.
        """
        try:
            # Attempt to load CSV without headers
            self.data = pd.read_csv(input_file_path, header=None)
            # If the first row contains only strings, assume it contains headers
            if all(isinstance(x, str) for x in self.data.iloc[0]):
                self.logger.info(
                    "Detected headers in the first row; re-reading CSV with headers."
                )
                self.data = pd.read_csv(input_file_path)
        except OSError as e:
            self.logger.exception(f"Failed to load CSV file: {input_file_path}")
            raise e
