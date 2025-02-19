"""Defines the AbstractDataloader abstract base class.

This module provides a template for creating dataloader classes that supply DataLoader instances
for training, validation, test, and cross-validation folds. It also defines methods for retrieving
feature, covariate, and target labels, as well as any skipped data.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader


class AbstractDataloader(ABC):
    """
    Abstract base class for dataloaders.

    Subclasses must implement methods to provide DataLoader instances for training,
    validation, test, and cross-validation, as well as methods to retrieve dataset metadata.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the AbstractDataLoader.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training data.
        """

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation data.
        """

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for test data.
        """

    @abstractmethod
    def fold_dataloader(self, fold: int) -> tuple[DataLoader, DataLoader]:
        """
        Get the DataLoader for training and validation data for a given fold.

        This method is used in cross-validation to split data into training and validation sets.

        Args:
            fold (int): Fold number.

        Returns:
            Tuple[DataLoader, DataLoader]: DataLoaders for training and validation data.
        """

    @abstractmethod
    def get_feature_labels(self) -> list[str]:
        """
        Get the labels of the features in the dataset.

        Returns:
            List[str]: List of feature labels.
        """

    @abstractmethod
    def get_covariate_labels(self) -> list[str]:
        """
        Get the labels of the covariates in the dataset.

        Returns:
            List[str]: List of covariate labels.
        """

    @abstractmethod
    def get_target_labels(self) -> list[str]:
        """
        Get the labels of the targets in the dataset.

        Returns:
            List[str]: List of target labels.
        """

    @abstractmethod
    def get_skipped_data(self) -> pd.DataFrame:
        """
        Get the skipped data as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the skipped columns.
        """
