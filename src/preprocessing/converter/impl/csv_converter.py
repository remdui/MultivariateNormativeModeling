"""Data converter for CSV files."""

import pandas as pd

from entities.log_manager import LogManager
from preprocessing.converter.abstract_data_converter import AbstractDataConverter


class CSVConverter(AbstractDataConverter):
    """Data converter for CSV Files."""

    def __init__(self) -> None:
        """Initialize the CSVConverter class."""
        super().__init__(LogManager.get_logger(__name__))

    def _load(self, input_file_path: str) -> None:
        """Load the CSV file."""
        try:
            self.data = pd.read_csv(input_file_path, header=None)
            # Check if the first row contains only strings (indicating possible headers) and re-read with headers if so
            if all(isinstance(x, str) for x in self.data.iloc[0]):
                self.logger.info(
                    "Detected headers in the first row; re-reading with headers."
                )
                self.data = pd.read_csv(input_file_path)
        except OSError as e:
            self.logger.exception(f"Failed to load {input_file_path}")
            raise e
