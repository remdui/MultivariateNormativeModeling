"""Factory method to get the loss function based on config."""

from data.abstract_dataloader import AbstractDataloader
from data.impl.image_data.image_2D_dataloader import Image2DDataLoader
from data.impl.image_data.image_3D_dataloader import Image3DDataLoader
from data.impl.tabular_data.tabular_dataloader import TabularDataloader


def get_dataloader(data_type: str) -> AbstractDataloader:
    """Factory method to get the loss function based on config."""
    if data_type == "tabular":
        return TabularDataloader()
    if data_type == "image2d":
        return Image2DDataLoader()
    if data_type == "image3d":
        return Image3DDataLoader()
    raise ValueError(f"Unknown data type: {data_type}")
