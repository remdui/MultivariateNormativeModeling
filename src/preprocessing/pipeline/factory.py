"""Factory function to create the appropriate preprocessing pipeline."""

from preprocessing.pipeline.abstract_pipeline import AbstractPreprocessingPipeline
from preprocessing.pipeline.impl.image_pipeline import ImagePreprocessingPipeline
from preprocessing.pipeline.impl.tabular_pipeline import TabularPreprocessingPipeline


def create_preprocessing_pipeline(data_type: str) -> AbstractPreprocessingPipeline:
    """Factory function to create the appropriate preprocessing pipeline."""
    if data_type == "tabular":
        return TabularPreprocessingPipeline()
    if data_type in {"image2d", "image3d"}:
        return ImagePreprocessingPipeline()
    raise ValueError(f"Unsupported data type: {data_type}")
