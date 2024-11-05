"""Abstract base class for preprocessing pipelines."""

from abc import ABC, abstractmethod
from logging import Logger

import pandas as pd

from entities.properties import Properties
from preprocessing.preprocessor.abstract_preprocessor import AbstractPreprocessor
from preprocessing.preprocessor.factory import get_preprocessor


class AbstractPreprocessingPipeline(ABC):
    """Abstract base class for preprocessing pipelines."""

    def __init__(self, logger: Logger) -> None:
        """Initialize the preprocessing pipeline and configure preprocessors."""
        self.logger = logger
        self.logger.info("Initializing Preprocessing Pipeline")
        self.properties = Properties.get_instance()
        self.preprocessors: list[AbstractPreprocessor] = []
        self.data: pd.DataFrame = pd.DataFrame()
        self.__init_preprocessors()

    def __init_preprocessors(self) -> None:
        """Initialize preprocessors based on configuration in properties."""
        for preprocessor_config in self.properties.dataset.preprocessors:
            preprocessor_name = preprocessor_config.name
            preprocessor_params = preprocessor_config.params or {}
            try:
                preprocessor_instance = get_preprocessor(
                    preprocessor_name, **preprocessor_params
                )
                self.preprocessors.append(preprocessor_instance)
                self.logger.info(f"Added preprocessor: {preprocessor_name}")
            except ValueError as e:
                self.logger.error(str(e))
                raise

    def run(self) -> None:
        """Run the preprocessing pipeline for the specific data type."""
        input_data = self.properties.dataset.input_data
        data_dir = self.properties.system.data_dir
        self.execute_pipeline(input_data, data_dir)

    @abstractmethod
    def execute_pipeline(self, input_data: str, data_dir: str) -> None:
        """Abstract method to execute the pipeline for a specific data type."""
