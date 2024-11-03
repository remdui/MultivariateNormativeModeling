"""Defines the AbstractDataset abstract base class."""

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class AbstractDataset(Dataset, ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the dataset."""
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """Retrieve an item from the dataset at the given index."""

    def get_num_samples(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self)

    def get_num_features(self) -> int:
        """Returns the number of features in the dataset.

        Returns:
            int: Number of features (excluding labels or covariates).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_num_covariates(self) -> int:
        """Returns the number of covariates in the dataset.

        Returns:
            int: Number of covariates.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
