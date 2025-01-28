"""Factory module for creating analysis engine instances."""

from typing import Any

from analysis.engine.abstract_analysis_engine import AbstractAnalysisEngine
from analysis.engine.impl.tabular_analysis_engine import TabularAnalysisEngine

# Mapping for available engines
ENGINE_MAPPING: dict[str, type[AbstractAnalysisEngine]] = {
    "tabular": TabularAnalysisEngine,
}


def create_analysis_engine(
    data_type: str, *args: Any, **kwargs: Any
) -> AbstractAnalysisEngine:
    """Factory function to create the appropriate analysis engine.

    Args:
        data_type (str): The type of data to be processed (e.g., 'tabular', 'image2d').
        *args: Additional arguments for the engine.
        **kwargs: Additional parameters for the engine.

    Returns:
        AbstractAnalysisEngine: An instance of the requested engine.

    Raises:
        ValueError: If the data type is not supported.
    """
    engine_class = ENGINE_MAPPING.get(data_type)
    if not engine_class:
        raise ValueError(f"Unsupported data type: {data_type}")
    return engine_class(*args, **kwargs)
