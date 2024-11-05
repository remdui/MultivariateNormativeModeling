"""Image3DDataset to process 3D image data."""

from typing import Any

from data.abstract_dataset import AbstractDataset


class Image3DDataset(AbstractDataset):
    """Image3DDataset to process 3D image data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Image3DDataset."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Get the length of the dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple:
        """Get the item at the given index."""
        raise NotImplementedError
