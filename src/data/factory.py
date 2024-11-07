"""Factory module for creating data loader instances."""

from data.abstract_dataloader import AbstractDataloader
from data.impl.image_data.image_2D_dataloader import Image2DDataLoader
from data.impl.image_data.image_3D_dataloader import Image3DDataLoader
from data.impl.tabular_data.tabular_dataloader import TabularDataloader

# Mapping for available data loaders
DATALOADER_MAPPING: dict[str, type[AbstractDataloader]] = {
    "tabular": TabularDataloader,
    "image2d": Image2DDataLoader,
    "image3d": Image3DDataLoader,
}


def get_dataloader(data_type: str) -> AbstractDataloader:
    """Factory method to get the data loader based on config.

    Args:
        data_type (str): The type of data (e.g., 'tabular', 'image2d', 'image3d').

    Returns:
        AbstractDataloader: An instance of the specified data loader.

    Raises:
        ValueError: If the data type is not supported.
    """
    dataloader_class = DATALOADER_MAPPING.get(data_type.lower())
    if not dataloader_class:
        raise ValueError(f"Unknown data type: {data_type}")
    return dataloader_class()
