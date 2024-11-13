"""Module to perform data exploration."""

from typing import Any

from analysis.exploration.abstract_data_exploration import AbstractDataExploration
from entities.log_manager import LogManager


class TabularDataExploration(AbstractDataExploration):
    """Class to perform data exploration."""

    def __init__(self, data: Any) -> None:
        """Initialize the DataExploration object."""
        super().__init__(data, LogManager.get_logger(__name__))
        self.data = data

    def run(self) -> None:
        """Run the data exploration pipeline."""
        self.logger.info("Starting Data Exploration.")

        self.logger.info(f"Number of samples: {self.data.get_num_samples()}")
        self.logger.info(f"Number of features: {self.data.get_num_features()}")
