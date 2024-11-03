"""Factory method to get the loss function based on config."""

from data.abstract_dataloader import AbstractDataloader
from data.impl.tabular_data.tabular_dataloader import TabularDataloader


def get_dataloader(data_type: str) -> AbstractDataloader:
    """Factory method to get the loss function based on config."""
    if data_type == "tabular":
        return TabularDataloader()
    raise ValueError(f"Unknown data type: {data_type}")
