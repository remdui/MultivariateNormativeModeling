"""Factory module for creating preprocessing pipeline instances."""

from typing import Any

from preprocessing.pipeline.abstract_pipeline import AbstractPreprocessingPipeline
from preprocessing.pipeline.impl.image_pipeline import ImagePreprocessingPipeline
from preprocessing.pipeline.impl.tabular_pipeline import TabularPreprocessingPipeline

# Type alias for pipeline classes
PipelineClass = type[AbstractPreprocessingPipeline]

# Mapping for available pipelines (private)
_PIPELINE_MAPPING: dict[str, PipelineClass] = {
    "tabular": TabularPreprocessingPipeline,
    "image2d": ImagePreprocessingPipeline,
    "image3d": ImagePreprocessingPipeline,
}


def create_preprocessing_pipeline(
    data_type: str, *args: Any, **kwargs: Any
) -> AbstractPreprocessingPipeline:
    """
    Factory function to create the appropriate preprocessing pipeline.

    Args:
        data_type (str): The type of data to be processed (e.g., 'tabular', 'image2d').
        *args: Additional positional arguments for the pipeline's constructor.
        **kwargs: Additional keyword arguments for the pipeline's constructor.

    Returns:
        AbstractPreprocessingPipeline: An instance of the requested preprocessing pipeline.

    Raises:
        ValueError: If the specified data type is not supported.
    """
    pipeline_class = _PIPELINE_MAPPING.get(data_type)
    if pipeline_class is None:
        raise ValueError(f"Unsupported data type: {data_type}")
    return pipeline_class(*args, **kwargs)
