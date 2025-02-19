"""Factory module for creating analysis engine instances."""

from typing import Any

from analysis.engine.abstract_analysis_engine import AbstractAnalysisEngine
from analysis.engine.impl.tabular_analysis_engine import TabularAnalysisEngine

# Type alias for engine classes (subclasses of AbstractAnalysisEngine)
EngineClass = type[AbstractAnalysisEngine]

# Mapping for available engines (private)
_ENGINE_MAPPING: dict[str, EngineClass] = {
    "tabular": TabularAnalysisEngine,
}


def create_analysis_engine(
    data_type: str, *args: Any, **kwargs: Any
) -> AbstractAnalysisEngine:
    """
    Factory function to create the appropriate analysis engine based on configuration.

    Args:
        data_type (str): The type of data to be processed (e.g., 'tabular').
                         The lookup is case-insensitive.
        *args: Additional positional arguments for the engine's constructor.
        **kwargs: Additional keyword arguments for the engine's constructor.

    Returns:
        AbstractAnalysisEngine: An instance of the requested analysis engine.

    Raises:
        ValueError: If the data type is not supported.
    """
    engine_class = _ENGINE_MAPPING.get(data_type.lower())
    if engine_class is None:
        raise ValueError(f"Unsupported data type: {data_type}")
    return engine_class(*args, **kwargs)
