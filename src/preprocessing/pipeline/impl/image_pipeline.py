"""Image data preprocessing pipeline implementation.

This module defines a pipeline for processing image data. Currently, the pipeline is not implemented;
attempts to run it will result in an ImagePipelineNotImplementedError.
"""

from entities.log_manager import LogManager
from preprocessing.pipeline.abstract_pipeline import AbstractPreprocessingPipeline


class ImagePipelineNotImplementedError(NotImplementedError):
    """Exception raised when the image preprocessing pipeline is not implemented."""


class ImagePreprocessingPipeline(AbstractPreprocessingPipeline):
    """
    Pipeline for processing image data.

    This pipeline is intended for preprocessing image data but is not yet implemented.
    """

    def __init__(self) -> None:
        """
        Initialize the image preprocessing pipeline.

        Sets up logging and other configuration via the abstract pipeline.
        """
        super().__init__(logger=LogManager.get_logger(__name__))

    def _execute_pipeline(self) -> None:
        """
        Execute the image data preprocessing pipeline.

        Raises:
            ImagePipelineNotImplementedError: Always raised since the image pipeline is not implemented.
        """
        self.logger.warning("Image data pipeline is not implemented yet.")
        raise ImagePipelineNotImplementedError(
            "Image data pipeline is not implemented yet."
        )
