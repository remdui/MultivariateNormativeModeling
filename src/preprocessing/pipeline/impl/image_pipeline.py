"""Image data preprocessing pipeline implementation."""

from entities.log_manager import LogManager
from preprocessing.pipeline.abstract_pipeline import AbstractPreprocessingPipeline


class ImagePreprocessingPipeline(AbstractPreprocessingPipeline):
    """Pipeline for processing image data."""

    def __init__(self) -> None:
        """Initialize the image preprocessing pipeline."""
        super().__init__(LogManager.get_logger(__name__))

    def execute_pipeline(self, input_data: str, data_dir: str) -> None:
        """Execute the image data preprocessing pipeline."""
        self.logger.warning("Image data pipeline is not implemented yet.")
        raise NotImplementedError("Image data pipeline is not implemented yet.")
