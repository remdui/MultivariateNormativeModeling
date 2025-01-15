"""Data converter for RDS files."""

import pyreadr  # type: ignore

from entities.log_manager import LogManager
from preprocessing.converter.abstract_data_converter import AbstractDataConverter


class RDSConverter(AbstractDataConverter):
    """Data converter for RDS Files."""

    def __init__(self) -> None:
        """Initialize the RDSConverter class."""
        super().__init__(LogManager.get_logger(__name__))

    def _load(self, input_file_path: str) -> None:
        """Load the RDS file."""
        try:
            result = pyreadr.read_r(str(input_file_path))
            # Assuming there's only one dataframe in the RDS file
            self.data = next(iter(result.values()))
            self.logger.info(self.data.head())
            self.logger.info(self.data.columns)
            self.logger.info(self.data.dtypes)
        except OSError as e:
            self.logger.exception(f"Failed to load {input_file_path}")
            raise e
