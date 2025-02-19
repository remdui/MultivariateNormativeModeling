"""Data Analysis module to perform data analysis.

This module defines an abstract base class for data analysis engines. Subclasses
should implement the data exploration pipeline initialization and analysis methods.
"""

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any

from entities.log_manager import LogManager
from entities.properties import Properties


class AbstractAnalysisEngine(ABC):
    """
    Abstract base class for data analysis engines.

    Provides common properties and settings for performing data analysis, including
    plot settings and feature configuration loaded from the global Properties.
    """

    def __init__(self, logger: Logger = LogManager.get_logger(__name__)) -> None:
        """
        Initialize the data analysis engine.

        Args:
            logger (Logger): Logger instance for logging analysis events.
        """
        self.logger = logger
        self.properties = Properties.get_instance()
        self.plot_settings = self.properties.data_analysis.plots
        self.feature_settings = self.properties.data_analysis.features

    @abstractmethod
    def initialize_engine(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the data analysis pipeline.

        Subclasses must implement this method to set up the analysis engine.
        This may include loading data, configuring analysis parameters, or preparing
        visualization settings.
        """
