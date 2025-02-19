"""DataLoader for 2D image data.

This abstract class defines the interface for DataLoaders that handle 2D image datasets.
Subclasses must implement methods to provide DataLoader instances for training, validation,
and testing, as well as methods to retrieve dataset metadata such as feature, covariate,
and target labels, along with any skipped data.
"""

import pandas as pd
from torch.utils.data import DataLoader

from data.abstract_dataloader import AbstractDataloader


class Image2DDataLoader(AbstractDataloader):
    """
    Abstract DataLoader for 2D image data.

    Subclasses must implement methods to return DataLoaders for training, validation, and test splits,
    and to provide labels for features, covariates, and targets, as well as any skipped data.
    """

    def train_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader instance for training data.
        """
        raise NotImplementedError("Subclasses must implement train_dataloader().")

    def val_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader instance for validation data.
        """
        raise NotImplementedError("Subclasses must implement val_dataloader().")

    def test_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader instance for test data.
        """
        raise NotImplementedError("Subclasses must implement test_dataloader().")

    def fold_dataloader(self, fold: int) -> tuple[DataLoader, DataLoader]:
        """
        Get DataLoaders for training and validation datasets for a specified fold.

        Args:
            fold (int): The fold index for cross-validation.

        Returns:
            Tuple[DataLoader, DataLoader]: DataLoaders for training and validation data.
        """
        raise NotImplementedError("Subclasses must implement fold_dataloader().")

    def get_feature_labels(self) -> list[str]:
        """
        Retrieve the feature labels from the dataset.

        Returns:
            List[str]: List of feature column names.
        """
        raise NotImplementedError("Subclasses must implement get_feature_labels().")

    def get_covariate_labels(self) -> list[str]:
        """
        Retrieve the covariate labels from the dataset.

        Returns:
            List[str]: List of covariate column names.
        """
        raise NotImplementedError("Subclasses must implement get_covariate_labels().")

    def get_target_labels(self) -> list[str]:
        """
        Retrieve the target labels from the dataset.

        Returns:
            List[str]: List of target column names.
        """
        raise NotImplementedError("Subclasses must implement get_target_labels().")

    def get_skipped_data(self) -> pd.DataFrame:
        """
        Retrieve the skipped data as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing columns that were skipped during data processing.
        """
        raise NotImplementedError("Subclasses must implement get_skipped_data().")
