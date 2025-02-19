"""Image3DDataset to process 3D image data.

This abstract class serves as a template for datasets that handle 3D image data.
Subclasses must implement the initialization, length, and item retrieval methods.
"""

from typing import Any

from data.abstract_dataset import AbstractDataset


class Image3DDataset(AbstractDataset):
    """
    Abstract dataset class for processing 3D image data.

    Subclasses should implement the __init__, __len__, and __getitem__ methods
    to load and serve 3D image data samples.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the Image3DDataset.

        Args:
            *args: Positional arguments for dataset initialization.
            **kwargs: Keyword arguments for dataset initialization.
        """
        raise NotImplementedError("Subclasses must implement __init__ method.")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        raise NotImplementedError("Subclasses must implement __len__ method.")

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve the sample at the specified index.

        Args:
            idx (int): Index of the desired sample.

        Returns:
            Tuple: A tuple representing a data sample; the structure is defined by the subclass.
        """
        raise NotImplementedError("Subclasses must implement __getitem__ method.")
