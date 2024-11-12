"""Abstract base class for preprocessing pipelines."""

from abc import ABC, abstractmethod
from logging import Logger

from torchvision.transforms.v2 import Transform  # type: ignore

from entities.log_manager import LogManager
from entities.properties import Properties
from preprocessing.transform.factory import get_transform


class AbstractPreprocessingPipeline(ABC):
    """Abstract base class for preprocessing pipelines."""

    def __init__(self, logger: Logger = LogManager.get_logger(__name__)) -> None:
        """Initialize the preprocessing pipeline and configure transformers."""
        self.logger = logger
        self.logger.info("Initializing Preprocessing Pipeline")
        self.properties = Properties.get_instance()
        self.transforms: list[Transform] = []
        self.__init_transforms()

    def __init_transforms(self) -> None:
        """Initialize transforms based on configuration in properties."""
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
                    self.logger.error(str(e))
                    raise

    def run(self) -> None:
        """Run the preprocessing pipeline for the specific data type."""
        self._execute_pipeline()

    @abstractmethod
    def _execute_pipeline(self) -> None:
        """Abstract method to execute the pipeline for a specific data type."""
