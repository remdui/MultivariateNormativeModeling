"""Data Analysis module to perform data analysis."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any

from entities.log_manager import LogManager
from entities.properties import Properties


class AbstractAnalysisEngine(ABC):
    """Class to perform data analysis."""

    def __init__(self, logger: Logger = LogManager.get_logger(__name__)) -> None:
        """Initialize the DataAnalysis object."""
        self.logger = logger
        self.properties = Properties.get_instance()

        # Get the data analysis plot settings
        self.plot_settings = self.properties.data_analysis.plots
        self.feature_settings = self.properties.data_analysis.features

    @abstractmethod
    def initialize_engine(self, *args: Any, **kwargs: Any) -> None:
        """Initialize data exploration pipeline."""

    @abstractmethod
    def calculate_reconstruction_mse(self) -> float:
        """Calculate mse of reconstruction."""

    @abstractmethod
    def calculate_reconstruction_r2(self) -> float:
        """Calculate r2 of reconstruction."""
