"""FreeSurferDataloader class."""

import torch
from torch.utils.data import DataLoader, random_split

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
        self.shuffle = self.properties.dataset.shuffle
        self.train_split = self.properties.dataset.train_split
        self.val_split = self.properties.dataset.val_split
        self.seed = self.properties.general.seed

        # Set up the dataloader
        self.__setup_file_paths()
        self.__initialize_datasets()

    def __setup_file_paths(self) -> None:
        """Set up the file paths for the training/validation, and test datasets."""
        if is_data_file(self.input_data):
            self.csv_path = get_processed_file_path(self.data_dir, self.input_data)
            self.test_csv_path = get_processed_file_path(
                self.data_dir, "test_" + self.input_data
            )
        else:
            raise ValueError(f"Invalid data format: {self.input_data}")

    def __initialize_datasets(self) -> None:
        """Set up the datasets for training, validation, and testing."""
        self.logger.info("Initializing TabularDataloader...")

        # Load the datasets
        dataset = self.__load__dataset(self.csv_path)
        self.test_dataset = self.__load__dataset(self.test_csv_path)

        # Get and log dataset sizes
        dataset_size = len(dataset)
        self.logger.info(f"Dataset size: {dataset_size}")
        test_dataset_size = len(self.test_dataset)
        self.logger.info(f"Test dataset size: {test_dataset_size}")

        # Calculate split sizes
        train_size = int(self.train_split * dataset_size)
        val_size = int(self.val_split * dataset_size)

        self.logger.info(f"Splitting dataset: {train_size} train, {val_size} val")

        # Split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.logger.info("Dataset split successfully into train and val sets.")

    def __load__dataset(self, file_path: str) -> TabularDataset:
        """Load the dataset from the provided file path."""
        try:
            dataset = TabularDataset(
                csv_file=str(file_path), covariates=self.covariates
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
            shuffle=False,
            num_workers=self.num_workers,
        )
