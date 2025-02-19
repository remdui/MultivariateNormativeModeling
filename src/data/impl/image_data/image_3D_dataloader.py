"""DataLoader for 3D image data.

This abstract class defines the interface for creating DataLoader instances for 3D image datasets.
Subclasses must implement methods for training, validation, testing, and cross-validation DataLoaders,
as well as methods for retrieving dataset metadata.
"""

import pandas as pd
from torch.utils.data import DataLoader

from data.abstract_dataloader import AbstractDataloader


class Image3DDataLoader(AbstractDataloader):
    """
    Abstract DataLoader for 3D image data.

    Subclasses must implement all methods to provide DataLoader instances and to retrieve
    feature, covariate, and target labels, as well as any skipped data.
    """

    def train_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training data.
        """
        raise NotImplementedError("Subclasses must implement train_dataloader()")

    def val_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        raise NotImplementedError("Subclasses must implement val_dataloader()")

    def test_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for test data.
        """
        raise NotImplementedError("Subclasses must implement test_dataloader()")

    def fold_dataloader(self, fold: int) -> tuple[DataLoader, DataLoader]:
        """
        Get DataLoaders for training and validation datasets for a given fold (for cross-validation).

        Args:
            fold (int): The fold index.

        Returns:
            Tuple[DataLoader, DataLoader]: DataLoaders for training and validation data.
        """
        raise NotImplementedError("Subclasses must implement fold_dataloader()")

    def get_feature_labels(self) -> list[str]:
        """
        Get the feature labels of the dataset.

        Returns:
            List[str]: List of feature column names.
        """
        raise NotImplementedError("Subclasses must implement get_feature_labels()")

    def get_covariate_labels(self) -> list[str]:
        """
        Get the covariate labels of the dataset.

        Returns:
            List[str]: List of covariate column names.
        """
        raise NotImplementedError("Subclasses must implement get_covariate_labels()")

    def get_target_labels(self) -> list[str]:
        """
        Get the target labels of the dataset.

        Returns:
            List[str]: List of target column names.
        """
        raise NotImplementedError("Subclasses must implement get_target_labels()")

    def get_skipped_data(self) -> pd.DataFrame:
        """
        Get the skipped data as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the skipped columns.
        """
        raise NotImplementedError("Subclasses must implement get_skipped_data()")
