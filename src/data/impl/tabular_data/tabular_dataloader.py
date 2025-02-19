"""TabularDataloader for loading tabular datasets.

This class implements the AbstractDataloader interface for tabular data. It loads processed
datasets, sets up training, validation, and test splits, and supports cross-validation.
"""

from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold  # type: ignore
from torch.utils.data import DataLoader, Subset, random_split

from data.abstract_dataloader import AbstractDataloader
from data.impl.tabular_data.tabular_dataset import TabularDataset
from entities.log_manager import LogManager
from entities.properties import Properties
from util.file_utils import get_processed_file_path, is_data_file


class TabularDataloader(AbstractDataloader):
    """
    Dataloader for Tabular datasets.

    Loads processed tabular data, splits it into training, validation, and test sets,
    and optionally sets up cross-validation folds.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the TabularDataloader using configuration properties.

        Retrieves paths, batch size, and other parameters from the global Properties.
        Sets up the datasets and cross-validation indices (if enabled).
        """
        super().__init__(*args, **kwargs)
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()

        # Retrieve configuration settings.
        self.data_dir = self.properties.system.data_dir
        self.input_data = self.properties.dataset.input_data
        self.batch_size = self.properties.train.batch_size
        self.num_workers = self.properties.system.num_workers
        self.covariates = self.properties.dataset.covariates
        self.skipped_covariates = self.properties.dataset.skipped_covariates
        self.targets = self.properties.dataset.targets
        self.pin_memory = self.properties.dataset.pin_memory
        self.shuffle = self.properties.dataset.shuffle
        self.train_split = self.properties.dataset.train_split
        self.val_split = self.properties.dataset.val_split
        self.seed = self.properties.general.seed

        self.__setup_file_paths()
        self.__initialize_datasets()
        self.__initialize_cross_validation()

    def __setup_file_paths(self) -> None:
        """
        Set up file paths for processed training/validation and test datasets.

        Uses get_processed_file_path() to derive paths. Raises ValueError if the input
        data format is invalid.
        """
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
        """
        Load and initialize datasets for training, validation, and testing.

        Loads the training and test datasets from their processed paths.
        If cross-validation is not enabled, splits the training data into training and validation sets.
        """
        self.logger.info("Initializing TabularDataloader...")

        train_dataset = self.__load__dataset(self.train_data_path)
        self.test_dataset = self.__load__dataset(self.test_data_path)

        self.logger.info(
            f"Train dataset: {len(train_dataset)} samples, {train_dataset.get_num_features()} features"
        )
        self.logger.info(
            f"Test dataset: {len(self.test_dataset)} samples, {self.test_dataset.get_num_features()} features"
        )

        self.logger.debug(f"Train dataset head:\n{train_dataset.data.head()}")
        self.logger.debug(f"Test dataset head:\n{self.test_dataset.data.head()}")

        if self.properties.train.cross_validation:
            self.train_dataset = train_dataset
        else:
            self.__split_train_val(train_dataset)

    def __load__dataset(self, file_path: str) -> TabularDataset:
        """
        Load a TabularDataset from the specified file path.

        Args:
            file_path (str): Path to the processed dataset file.

        Returns:
            TabularDataset: The loaded dataset.

        Raises:
            Exception: Propagates any exception encountered during loading.
        """
        try:
            dataset = TabularDataset(file_path=str(file_path))
            self.logger.info(f"Dataset loaded from {file_path}")
        except Exception as e:
            self.logger.exception("Failed to initialize the tabular dataset.")
            raise e

        self.features = dataset.features
        return dataset

    def train_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training data.

        Raises:
            ValueError: If the training dataset is not initialized or empty.
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
        """
        Get the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation data.

        Raises:
            ValueError: If the validation dataset is not initialized or empty.
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
        """
        Get the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for test data.

        Raises:
            ValueError: If the test dataset is not initialized or empty.
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
        """
        Get DataLoaders for training and validation data for a specific cross-validation fold.

        Args:
            fold (int): The fold index.

        Returns:
            tuple[DataLoader, DataLoader]: DataLoaders for training and validation.

        Raises:
            ValueError: If cross-validation is not enabled or fold indices are uninitialized.
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
        """
        Initialize cross-validation folds if enabled.

        Uses the method specified in configuration (kfold, stratifiedkfold, or groupkfold)
        to split the training dataset into folds.
        """
        if self.properties.train.cross_validation:
            method = self.properties.train.cross_validation_method
            num_folds = self.properties.train.cross_validation_folds
            self.logger.info(
                f"Cross-validation enabled: {num_folds} folds using {method}"
            )
            if method == "kfold":
                splitter = KFold(
                    n_splits=num_folds, shuffle=True, random_state=self.seed
                )
            elif method == "stratifiedkfold":
                splitter = StratifiedKFold(
                    n_splits=num_folds, shuffle=True, random_state=self.seed
                )
            elif method == "groupkfold":
                splitter = GroupKFold(n_splits=num_folds)
            else:
                raise ValueError(f"Invalid cross-validation method: {method}")

            self.train_val_indices = []
            for train_idx, val_idx in splitter.split(self.train_dataset):
                self.train_val_indices.append((train_idx, val_idx))

    def __split_train_val(self, train_dataset: Any) -> None:
        """
        Split the training dataset into training and validation sets.

        Uses random_split with a fixed seed to ensure reproducibility.

        Args:
            train_dataset (Any): The training dataset to split.
        """
        self.train_dataset, self.val_dataset = random_split(
            train_dataset,
            [self.train_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.logger.info(
            f"Splitting train dataset: {len(self.train_dataset)} train samples, {len(self.val_dataset)} validation samples"
        )

    def get_feature_labels(self) -> list[str]:
        """
        Get the feature labels of the dataset.

        Returns:
            list[str]: List of feature column names.
        """
        return self.features

    def get_covariate_labels(self) -> list[str]:
        """
        Get the covariate labels of the dataset.

        Returns:
            list[str]: List of covariate column names (excluding skipped covariates).
        """
        return [item for item in self.covariates if item not in self.skipped_covariates]

    def get_target_labels(self) -> list[str]:
        """
        Get the target labels of the dataset.

        Returns:
            list[str]: List of target column names.
        """
        return self.targets

    def get_skipped_data(self) -> pd.DataFrame:
        """
        Retrieve the skipped data as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the skipped columns.
        """
        return self.test_dataset.get_skipped_data()
