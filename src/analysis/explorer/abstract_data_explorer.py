"""Module to perform data exploration."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any

from entities.log_manager import LogManager
from entities.properties import Properties


class AbstractDataExplorer(ABC):
    """Class to perform data exploration."""

    def __init__(
        self, data: Any, logger: Logger = LogManager.get_logger(__name__)
    ) -> None:
        """Initialize the DataExploration object."""
        self.data = data
        self.logger = logger
        self.properties = Properties.get_instance()
        self.data_exploration = self.properties.data_analysis.data_exploration

    @abstractmethod
    def run(self) -> None:
        """Run the data exploration pipeline."""
