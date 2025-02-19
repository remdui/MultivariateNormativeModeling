"""Data converter for RDS files.

This module defines the RDSConverter class, which extends the AbstractDataConverter to load
data from RDS files using the pyreadr library. It assumes that the RDS file contains a single
DataFrame. If no DataFrame is found, a StopIteration error will be raised.
"""

import pyreadr  # type: ignore

from entities.log_manager import LogManager
from preprocessing.converter.abstract_data_converter import AbstractDataConverter


class RDSConverter(AbstractDataConverter):
    """
    Data converter for RDS files.

    Loads an RDS file and extracts its contained DataFrame. The converter assumes that the file
    includes only one DataFrame.
    """

    def __init__(self) -> None:
        """
        Initialize the RDSConverter.

        Sets up the logger for the converter.
        """
        super().__init__(LogManager.get_logger(__name__))

    def _load(self, input_file_path: str) -> None:
        """
        Load data from an RDS file.

        Reads the RDS file using pyreadr and assigns the first DataFrame found to self.data.

        Args:
            input_file_path (str): The file path of the RDS file to load.

        Raises:
            OSError: If the file cannot be read.
            StopIteration: If no DataFrame is found in the RDS file.
        """
        try:
            result = pyreadr.read_r(str(input_file_path))
            # Assume the RDS file contains only one DataFrame.
            self.data = next(iter(result.values()))
        except OSError as e:
            self.logger.exception(f"Failed to load RDS file: {input_file_path}")
            raise e
