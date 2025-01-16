"""Factory module for creating transform instances."""

from typing import Any

from torchvision.transforms.v2 import Grayscale, Normalize, Transform  # type: ignore

from preprocessing.transform.impl.age_filter import AgeFilterTransform
from preprocessing.transform.impl.data_cleaning import DataCleaningTransform
from preprocessing.transform.impl.noise import NoiseTransform
from preprocessing.transform.impl.normalization import NormalizationTransform
from preprocessing.transform.impl.sample_limit import SampleLimitTransform
from preprocessing.transform.impl.sex_filter import SexFilterTransform
from preprocessing.transform.impl.site_filter import SiteFilterTransform
from preprocessing.transform.impl.wave_filter import WaveFilterTransform

# Mapping for available transforms, including custom and torchvision transforms
TRANSFORM_MAPPING: dict[str, type[Any]] = {
    # Custom transforms
    "NormalizationTransform": NormalizationTransform,
    "DataCleaningTransform": DataCleaningTransform,
    "NoiseTransform": NoiseTransform,
    "SiteFilterTransform": SiteFilterTransform,
    "AgeFilterTransform": AgeFilterTransform,
    "SexFilterTransform": SexFilterTransform,
    "WaveFilterTransform": WaveFilterTransform,
    "SampleLimitTransform": SampleLimitTransform,
    # Torchvision transforms
    "Grayscale": Grayscale,
    "Normalize": Normalize,
}


def get_transform(name: str, *args: Any, **kwargs: Any) -> Transform:
    """Factory method to get a transform instance by name.

    Args:
        name (str): The name of the transform.
        *args: Additional arguments for the transform.
        **kwargs: Additional parameters for the transform.

    Returns:
        Transform: An instance of the requested transform.

    Raises:
        ValueError: If the transform name is not found in the mapping.
    """
    transform_class = TRANSFORM_MAPPING.get(name)
    if not transform_class:
        raise ValueError(f"Unknown transform: {name}")
    return transform_class(*args, **kwargs)
