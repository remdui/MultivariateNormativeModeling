"""Factory module for creating preprocessing pipeline instances."""

from preprocessing.pipeline.abstract_pipeline import AbstractPreprocessingPipeline
from preprocessing.pipeline.impl.image_pipeline import ImagePreprocessingPipeline
from preprocessing.pipeline.impl.tabular_pipeline import TabularPreprocessingPipeline

# Mapping for available pipelines
PIPELINE_MAPPING: dict[str, type[AbstractPreprocessingPipeline]] = {
    "tabular": TabularPreprocessingPipeline,
    "image2d": ImagePreprocessingPipeline,
    "image3d": ImagePreprocessingPipeline,
}


def create_preprocessing_pipeline(data_type: str) -> AbstractPreprocessingPipeline:
    """Factory function to create the appropriate preprocessing pipeline.

    Args:
        data_type (str): The type of data to be processed (e.g., 'tabular', 'image2d').

    Returns:
        AbstractPreprocessingPipeline: An instance of the requested pipeline.

    Raises:
        ValueError: If the data type is not supported.
    """
    pipeline_class = PIPELINE_MAPPING.get(data_type)
    if not pipeline_class:
        raise ValueError(f"Unsupported data type: {data_type}")
    return pipeline_class()
