"""Module to perform data exploration."""

from typing import Any

from analysis.explorer.abstract_data_explorer import AbstractDataExplorer
from entities.log_manager import LogManager


class TabularDataExplorer(AbstractDataExplorer):
    """Class to perform data exploration."""

    def __init__(self, data: Any) -> None:
        """Initialize the DataExploration object."""
        super().__init__(data, LogManager.get_logger(__name__))
        self.data = data

    def run(self) -> None:
        """Run the data exploration pipeline."""
        self.logger.info("Starting Data Explorer.")

        self.logger.info(f"Number of samples: {self.data.get_num_samples()}")
        self.logger.info(f"Number of features: {self.data.get_num_features()}")
