"""Defines the AbstractDataloader abstract base class."""

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class AbstractDataloader(ABC):
    """Abstract base class for dataloaders."""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Get the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training data.
        """

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Get the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation data.
        """

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Get the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for test data.
        """

    @abstractmethod
    def fold_dataloader(self, fold: int) -> tuple[DataLoader, DataLoader]:
        """Get the DataLoader for the training and validation data for a given fold.

        This is used for cross-validation methods.

        Args:
            fold (int): Fold number.

        Returns:
            tuple[DataLoader, DataLoader]: DataLoader for training and validation data.
        """
