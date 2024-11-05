"""Data converter for RDS files to CSV Files."""

import pandas as pd
import pyreadr  # type: ignore

from entities.log_manager import LogManager
from entities.properties import Properties
from preprocessing.converter.abstract_dataconverter import AbstractDataConverter


class RDSToCSVDataConverter(AbstractDataConverter):
    """Data converter for RDS files to CSV Files."""

    def __init__(self) -> None:
        """Initialize the RDSToCSVDataConverter class."""
        properties = Properties.get_instance()
        self.data_dir = properties.system.data_dir
        self.logger = LogManager.get_logger(__name__)

    def convert(self, input_file_path: str) -> pd.DataFrame:
        """Convert the RDS file to CSV format."""
        self.logger.info(f"Converting {input_file_path}")
        try:
            result = pyreadr.read_r(str(input_file_path))
            # Assuming there's only one dataframe in the RDS file
            data = next(iter(result.values()))
            self.logger.info(f"Successfully converted {input_file_path}")
            return data
        except OSError as e:
            self.logger.exception(f"Failed to convert {input_file_path}")
            raise e
