"""Factory module for creating preprocessor instances."""

from typing import Any

from preprocessing.preprocessor.abstract_preprocessor import AbstractPreprocessor
from preprocessing.preprocessor.impl.data_cleaning import DataCleaningPreprocessor
from preprocessing.preprocessor.impl.normalization import NormalizationPreprocessor

# Mapping for available preprocessors
PREPROCESSOR_MAPPING: dict[str, type[AbstractPreprocessor]] = {
    "NormalizationPreprocessor": NormalizationPreprocessor,
    "DataCleaningPreprocessor": DataCleaningPreprocessor,
}


def get_preprocessor(name: str, **params: Any) -> AbstractPreprocessor:
    """Factory method to get a preprocessor instance by name.

    Args:
        name (str): The name of the preprocessor.
        **params: Additional parameters for the preprocessor.

    Returns:
        AbstractPreprocessor: An instance of the requested preprocessor.

    Raises:
        ValueError: If the preprocessor name is not found in the mapping.
    """
    preprocessor_class = PREPROCESSOR_MAPPING.get(name)
    if not preprocessor_class:
        raise ValueError(f"Unknown preprocessor: {name}")
    return preprocessor_class(**params)
