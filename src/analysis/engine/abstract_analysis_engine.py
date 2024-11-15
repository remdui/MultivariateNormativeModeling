"""Data Analysis module to perform data analysis."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any

from analysis.explorer.exploration_phase import DataExplorationPhase
from entities.log_manager import LogManager
from entities.properties import Properties


class AbstractAnalysisEngine(ABC):
    """Class to perform data analysis."""

    def __init__(self, logger: Logger = LogManager.get_logger(__name__)) -> None:
        """Initialize the DataAnalysis object."""
        self.logger = logger
        self.properties = Properties.get_instance()

        # Get the data analysis settings
        self.visualization_settings = self.properties.data_analysis.visualization
        self.dimensionality_reduction_settings = (
            self.properties.data_analysis.dimensionality_reduction
        )

        # Initialize the data analysis modules
        self.__init_data_analysis_modules()

    def __init_data_analysis_modules(self) -> None:
        """Initialize and store data analysis properties as modules."""
        # Storing each analysis component's configuration for easy access
        self.data_exploration_module = self.properties.data_analysis.data_exploration
        self.reconstruction_analysis_module = (
            self.properties.data_analysis.reconstruction_analysis
        )
        self.latent_space_analysis_module = (
            self.properties.data_analysis.latent_space_analysis
        )

    @abstractmethod
    def run_data_exploration(self, phase: DataExplorationPhase, data: Any) -> None:
        """Run the data exploration pipeline for the specified phase."""

    @abstractmethod
    def run_reconstruction_analysis(self) -> None:
        """Run the reconstruction analysis pipeline."""

    @abstractmethod
    def run_latent_space_analysis(self) -> None:
        """Run the latent space analysis pipeline."""
