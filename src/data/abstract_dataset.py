"""Defines the AbstractDataset abstract base class.

This module provides a template for dataset classes. Subclasses must implement methods to
initialize the dataset, return its size, and access individual samples. Optionally, subclasses
can implement methods to report the number of features and covariates.
"""

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class AbstractDataset(Dataset, ABC):
    """
    Abstract base class for datasets.

    Subclasses must implement the __init__, __len__, and __getitem__ methods.
    Optionally, methods get_num_features() and get_num_covariates() can be implemented to
    provide additional dataset metadata.
    """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the dataset.

        Subclasses should load or generate data and set up any required attributes.
        """
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the size of the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            idx (int): Index of the desired sample.

        Returns:
            tuple: A tuple containing the sample data. The tuple's structure is defined by the subclass.
        """

    def get_num_samples(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: The dataset size, equivalent to len(self).
        """
        return len(self)

    def get_num_features(self) -> int:
        """
        Return the number of features in each sample.

        This method should be implemented by subclasses to report the feature dimension,
        excluding any labels or covariate information.

        Returns:
            int: Number of features.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement get_num_features().")

    def get_num_covariates(self) -> int:
        """
        Return the number of covariates in the dataset.

        Subclasses should implement this method if the dataset includes additional covariate
        information alongside the primary features.

        Returns:
            int: Number of covariates.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement get_num_covariates().")
