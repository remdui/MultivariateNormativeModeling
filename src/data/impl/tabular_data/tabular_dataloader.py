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

        if is_data_file(self.input_data):
            self.csv_path = get_processed_file_path(
                self.properties.system.data_dir, self.input_data
            )
        else:
            raise ValueError(f"Invalid data format: {self.input_data}")

        self.batch_size = self.properties.train.batch_size
        self.num_workers = self.properties.system.num_workers
        self.covariates = self.properties.dataset.covariates
        self.shuffle = self.properties.dataset.shuffle
        self.train_split = self.properties.dataset.train_split
        self.val_split = self.properties.dataset.val_split
        self.test_split = self.properties.dataset.test_split
        self.seed = self.properties.general.seed

        # Set up the dataloader
        self._setup()

    def _setup(self) -> None:
        """Set up the datasets for training, validation, and testing."""
        self.logger.info("Initializing TabularDataloader...")
        try:
            self.dataset = TabularDataset(
                csv_file=str(self.csv_path), covariates=self.covariates
            )
            self.logger.info(f"Dataset loaded from {self.csv_path}")
        except Exception as e:
            self.logger.exception("Failed to initialize the dataset.")
            raise e

        dataset_size = len(self.dataset)
        self.logger.info(f"Total dataset size: {dataset_size}")

        # Calculate split sizes
        train_size = int(self.train_split * dataset_size)
        val_size = int(self.val_split * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Handle rounding issues
        if test_size < 0:
            self.logger.warning("Adjusting test_size to 0 due to rounding issues.")
            test_size = 0
            val_size = dataset_size - train_size
        elif test_size == 0 and self.test_split > 0:
            self.logger.warning("Adjusting test_size to 1 to include test data.")
            test_size = 1
            val_size -= 1

        self.logger.info(
            f"Splitting dataset: {train_size} train, {val_size} val, {test_size} test"
        )

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.logger.info("Datasets split successfully.")

    def train_dataloader(self) -> DataLoader:
        """Get the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training data.

        Raises:
            ValueError: If the training dataset has not been initialized.
        """
        if self.train_dataset is None:
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
        if self.val_dataset is None:
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
        if self.test_dataset is None:
            self.logger.error("Test dataset not correctly initialized.")
            raise ValueError("Test dataset not correctly initialized.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
