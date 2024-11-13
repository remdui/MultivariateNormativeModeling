"""Data Analysis module to perform data analysis."""

from typing import Any

from analysis.abstract_data_analysis import AbstractDataAnalysis
from analysis.exploration.exploration_phase import DataExplorationPhase
from analysis.exploration.impl.tabular_data_exploration import TabularDataExploration
from entities.log_manager import LogManager


class TabularDataAnalysis(AbstractDataAnalysis):
    """Class to perform data analysis for tabular data."""

    def __init__(self) -> None:
        """Initialize the DataAnalysis object."""
        super().__init__(LogManager.get_logger(__name__))

    def run_data_exploration(self, phase: DataExplorationPhase, data: Any) -> None:
        """Run the data exploration pipeline for the specified phase."""
        self.logger.info(f"Running Data Exploration for {phase.name} phase.")

        # Retrieve phase configuration dynamically
        phase_config = self.data_exploration_module.phases.get(str(phase))

        if phase_config:
            TabularDataExploration(data).run()
        else:
            self.logger.warning(
                f"Data Exploration phase {phase.name} is not configured or disabled."
            )

    def run_reconstruction_analysis(self) -> None:
        """Run the reconstruction analysis pipeline."""

    def run_latent_space_analysis(self) -> None:
        """Run the latent space analysis pipeline."""
