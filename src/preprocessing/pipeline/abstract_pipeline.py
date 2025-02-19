"""Abstract base class for preprocessing pipelines.

This module defines an abstract pipeline for data preprocessing. It initializes a sequence
of transforms based on configuration settings (via Properties) and provides a common interface
to run the pipeline.
"""

from abc import ABC, abstractmethod
from logging import Logger

from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager
from entities.properties import Properties
from preprocessing.transform.factory import get_transform


class AbstractPreprocessingPipeline(ABC):
    """
    Abstract base class for preprocessing pipelines.

    This class initializes a list of preprocessing transforms based on the dataset configuration.
    Subclasses must implement the _execute_pipeline() method to define how the pipeline is executed.
    """

    def __init__(self, logger: Logger = LogManager.get_logger(__name__)) -> None:
        """
        Initialize the preprocessing pipeline and configure transforms.

        The pipeline retrieves its configuration from the Properties instance and constructs
        a list of transforms for preprocessing data.

        Args:
            logger (Logger): Logger instance for logging events.
        """
        self.logger = logger
        self.logger.info("Initializing Preprocessing Pipeline")
        self.properties = Properties.get_instance()
        self.transforms: list[Transform] = []
        self.__init_transforms()

    def __init_transforms(self) -> None:
        """
        Initialize preprocessing transforms based on configuration.

        This method iterates over the dataset's transform configuration and adds each transform
        with type "preprocessing" to the pipeline. If the transform cannot be instantiated,
        a ValueError (or a more specific custom exception, if available) is raised.
        """
        for transform_config in self.properties.dataset.transforms:
            if transform_config.type == "preprocessing":
                transform_name = transform_config.name
                transform_params = transform_config.params or {}
                try:
                    transform_instance = get_transform(
                        transform_name, **transform_params
                    )
                    self.transforms.append(transform_instance)
                    self.logger.info(f"Added transform: {transform_name}")
                except ValueError as e:
                    self.logger.error(
                        f"Error adding transform '{transform_name}': {str(e)}"
                    )
                    raise

    def run(self) -> None:
        """
        Execute the preprocessing pipeline.

        This method triggers the execution of the pipeline, which is defined by the
        _execute_pipeline() abstract method that must be implemented by subclasses.
        """
        self._execute_pipeline()

    @abstractmethod
    def _execute_pipeline(self) -> None:
        """
        Execute the pipeline for a specific data type.

        Subclasses must implement this method to define the actual data preprocessing steps.
        """
