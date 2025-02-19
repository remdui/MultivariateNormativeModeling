"""Factory module for creating data loader instances."""

from typing import Any

from data.abstract_dataloader import AbstractDataloader
from data.impl.image_data.image_2D_dataloader import Image2DDataLoader
from data.impl.image_data.image_3D_dataloader import Image3DDataLoader
from data.impl.tabular_data.tabular_dataloader import TabularDataloader

# Type alias for data loader classes (subclasses of AbstractDataloader)
DataLoaderClass = type[AbstractDataloader]

# Mapping for available data loaders (private)
_DATALOADER_MAPPING: dict[str, DataLoaderClass] = {
    "tabular": TabularDataloader,
    "image2d": Image2DDataLoader,
    "image3d": Image3DDataLoader,
}


def get_dataloader(data_type: str, *args: Any, **kwargs: Any) -> AbstractDataloader:
    """
    Factory method to create a data loader instance based on configuration.

    Args:
        data_type (str): The type of data (e.g., 'tabular', 'image2d', 'image3d').
                         The lookup is case-insensitive.
        *args: Additional positional arguments for the data loader's constructor.
        **kwargs: Additional keyword arguments for the data loader's constructor.

    Returns:
        AbstractDataloader: An instance of the specified data loader.

    Raises:
        ValueError: If the data type is not supported.
    """
    dataloader_class = _DATALOADER_MAPPING.get(data_type.lower())
    if dataloader_class is None:
        raise ValueError(f"Unknown data type: {data_type}")
    return dataloader_class(*args, **kwargs)
