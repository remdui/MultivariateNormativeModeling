"""FreeSurferDataloader class."""

from typing import Any

import torch
from sklearn.model_selection import KFold  # type: ignore
from torch.utils.data import DataLoader, Subset, random_split

from data.abstract_dataloader import AbstractDataloader
from data.impl.tabular_data.tabular_dataset import TabularDataset
from entities.log_manager import LogManager
from entities.properties import Properties
from util.file_utils import get_processed_file_path, is_data_file


class TabularDataloader(AbstractDataloader):
    """Dataloader for Tabular datasets."""

    def __init__(self) -> None:
        """Initialize the TabularDataloader using properties."""
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()

        # Access configuration directly from properties
        self.data_dir = self.properties.system.data_dir
        self.input_data = self.properties.dataset.input_data
        self.batch_size = self.properties.train.batch_size
        self.num_workers = self.properties.system.num_workers
        self.covariates = self.properties.dataset.covariates
        self.pin_memory = self.properties.dataset.pin_memory
        self.shuffle = self.properties.dataset.shuffle
        self.train_split = self.properties.dataset.train_split
        self.val_split = self.properties.dataset.val_split
        self.seed = self.properties.general.seed

        # Set up the dataloader
        self.__setup_file_paths()
        self.__initialize_datasets()
        self.__initialize_cross_validation()

    def __setup_file_paths(self) -> None:
        """Set up the file paths for the training/validation, and test datasets."""
        if is_data_file(self.input_data):
            self.train_data_path = get_processed_file_path(
                self.data_dir, self.input_data, "train"
            )
            self.test_data_path = get_processed_file_path(
                self.data_dir, self.input_data, "test"
            )
        else:
            raise ValueError(f"Invalid data format: {self.input_data}")

    def __initialize_datasets(self) -> None:
        """Set up the datasets for training, validation, and testing."""
        self.logger.info("Initializing TabularDataloader...")

        # Load the datasets
        train_dataset = self.__load__dataset(self.train_data_path)
        self.test_dataset = self.__load__dataset(self.test_data_path)

        # Get and log dataset sizes
        train_dataset_size = len(train_dataset)
        train_dataset_features = train_dataset.get_num_features()
        self.logger.info(
            f"Train dataset: {train_dataset_size} samples, {train_dataset_features} features"
        )

        test_dataset_size = len(self.test_dataset)
        test_dataset_features = self.test_dataset.get_num_features()
        self.logger.info(
            f"Test dataset: {test_dataset_size} samples, {test_dataset_features} features"
        )

        if self.properties.train.cross_validation:
            self.train_dataset = train_dataset
            self.val_dataset = None
        else:
            self.__split_train_val(train_dataset, train_dataset_size)

    def __load__dataset(self, file_path: str) -> TabularDataset:
        """Load the dataset from the provided file path."""
        try:
            dataset = TabularDataset(
                file_path=str(file_path), covariates=self.covariates
            )
            self.logger.info(f"Dataset loaded from {file_path}")
        except Exception as e:
            self.logger.exception("Failed to initialize the tabular dataset.")
            raise e

        return dataset

    def train_dataloader(self) -> DataLoader:
        """Get the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training data.

        Raises:
            ValueError: If the training dataset has not been initialized.
        """
        if self.train_dataset is None or len(self.train_dataset) == 0:
            self.logger.error("Train dataset not correctly initialized.")
            raise ValueError("Train dataset not correctly initialized.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation data.

        Raises:
            ValueError: If the validation dataset has not been initialized.
        """
        if self.val_dataset is None or len(self.val_dataset) == 0:
            self.logger.error("Validation dataset not correctly initialized.")
            raise ValueError("Validation dataset not correctly initialized.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the DataLoader for the test dataset.

        Returns:
            Optional[DataLoader]: DataLoader for test data, or None if test data is not available.

        Raises:
            ValueError: If the test dataset has not been initialized.
        """
        if self.test_dataset is None or len(self.test_dataset) == 0:
            self.logger.error("Test dataset not correctly initialized.")
            raise ValueError("Test dataset not correctly initialized.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def fold_dataloader(self, fold: int) -> tuple[DataLoader, DataLoader]:
        """Get the DataLoader for the training and validation datasets for a given fold.

        Args:
            fold (int): The fold index.

        Returns:
            tuple[DataLoader, DataLoader]: DataLoader for training and validation data.

        Raises:
            ValueError: If the training dataset has not been initialized.
        """
        if not self.properties.train.cross_validation:
            raise ValueError("Cross-validation not enabled.")

        if fold >= self.properties.train.cross_validation_folds:
            raise ValueError(
                f"Fold index {fold} out of range for {self.properties.train.cross_validation_folds}-fold cross-validation."
            )

        if self.train_val_indices is None:
            self.logger.error("Fold indices not correctly initialized.")
            raise ValueError("Fold indices not correctly initialized.")

        train_idx, val_idx = self.train_val_indices[fold]

        train_dataset = Subset(self.train_dataset, train_idx)
        val_dataset = Subset(self.train_dataset, val_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader

    def __initialize_cross_validation(self) -> None:
        """Initialize the cross-validation folds if enabled."""
        if self.properties.train.cross_validation:
            self.logger.info(
                f"Cross-validation enabled: splitting training data into {self.properties.train.cross_validation_folds} folds."
            )
            self.kfold = KFold(
                n_splits=self.properties.train.cross_validation_folds,
                shuffle=True,
                random_state=self.seed,
            )
            # Store the indices of the training and validation sets for each fold
            self.train_val_indices = []
            for train_idx, val_idx in self.kfold.split(self.train_dataset):
                self.train_val_indices.append((train_idx, val_idx))

    def __split_train_val(self, train_dataset: Any, train_dataset_size: int) -> None:
        """Split the training dataset into training and validation sets.

        Args:
            train_dataset (Any): The training dataset.
            train_dataset_size (int): The size of the training dataset.
        """
        # Calculate split sizes
        train_size = int(self.train_split * train_dataset_size)
        val_size = int(self.val_split * train_dataset_size)
        self.logger.info(
            f"Splitting train dataset: {train_size} train samples, {val_size} validation samples"
        )

        # Split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(  # type: ignore
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.logger.info("Dataset split successfully into train and validation sets.")
