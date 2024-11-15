"""Factory module for creating data analysis instances."""

from typing import Any

from analysis.engine.abstract_analysis_engine import AbstractAnalysisEngine
from analysis.engine.impl.tabular_analysis_engine import TabularAnalysisEngine

# Mapping for available data analysis types
ANALYSIS_MAPPING: dict[str, type[AbstractAnalysisEngine]] = {
    "tabular": TabularAnalysisEngine,
}


def create_data_analysis(
    data_type: str, *args: Any, **kwargs: Any
) -> AbstractAnalysisEngine:
    """Factory function to create the appropriate data analysis instance.

    Args:
        data_type (str): The type of data analysis to be performed (e.g., 'tabular', 'image2d', 'time_series').
        *args: Additional arguments for the analysis.
        **kwargs: Additional parameters for the analysis.

    Returns:
        AbstractDataAnalysis: An instance of the requested data analysis.

    Raises:
        ValueError: If the data type is not supported.
    """
    analysis_class = ANALYSIS_MAPPING.get(data_type)
    if not analysis_class:
        raise ValueError(f"Unsupported data type for analysis: {data_type}")
    return analysis_class(*args, **kwargs)
