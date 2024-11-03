"""Defines the AbstractDataloader abstract base class."""

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class AbstractDataloader(ABC):
    """Abstract base class for dataloaders."""

    @abstractmethod
    def _setup(self) -> None:
        """Set up the datasets for training, validation, and testing."""

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
