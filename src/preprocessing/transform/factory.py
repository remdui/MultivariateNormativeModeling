"""Factory module for creating transform instances."""

from typing import Any

from torchvision.transforms.v2 import Grayscale, Normalize, Transform  # type: ignore

from preprocessing.transform.impl.age_filter import AgeFilterTransform
from preprocessing.transform.impl.data_cleaning import DataCleaningTransform
from preprocessing.transform.impl.encoding import EncodingTransform
from preprocessing.transform.impl.noise import NoiseTransform
from preprocessing.transform.impl.sample_limit import SampleLimitTransform
from preprocessing.transform.impl.sex_filter import SexFilterTransform
from preprocessing.transform.impl.site_filter import SiteFilterTransform
from preprocessing.transform.impl.wave_filter import WaveFilterTransform

# Type alias for transform classes
TransformClass = type[Transform]

# Mapping for available transforms, including custom and torchvision transforms.
_TRANSFORM_MAPPING: dict[str, TransformClass] = {
    # Custom transforms
    "EncodingTransform": EncodingTransform,
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
    """
    Factory method to get a transform instance by name.

    Args:
        name (str): The name of the transform.
        *args: Positional arguments passed to the transform's constructor.
        **kwargs: Keyword arguments passed to the transform's constructor.

    Returns:
        Transform: An instance of the requested transform.

    Raises:
        ValueError: If the transform name is not found in the mapping.
    """
    transform_class = _TRANSFORM_MAPPING.get(name)
    if transform_class is None:
        raise ValueError(f"Unknown transform: {name}")
    return transform_class(*args, **kwargs)
