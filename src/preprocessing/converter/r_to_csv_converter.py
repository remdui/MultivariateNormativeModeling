"""Data converter for RDS files to CSV Files."""

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

    def convert(self, input_file_name: str, output_file_name: str) -> None:
        """Convert the RDS file to CSV format."""
        rds_file_path = f"{self.data_dir}/{input_file_name}"
        csv_file_path = f"{self.data_dir}/processed/{output_file_name}"

        self.logger.info(f"Converting {rds_file_path} to {csv_file_path}")
        try:
            result = pyreadr.read_r(str(rds_file_path))
            # Assuming there's only one dataframe in the RDS file
            df = next(iter(result.values()))
            df.to_csv(csv_file_path, index=False)
            self.logger.info(
                f"Successfully converted {rds_file_path} to {csv_file_path}"
            )
        except OSError as e:
            self.logger.exception(
                f"Failed to convert {rds_file_path} to {csv_file_path}"
            )
            raise e
