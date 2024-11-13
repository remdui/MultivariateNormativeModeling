"""Implementation of the tabular data preprocessing pipeline."""

import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split  # type: ignore

from entities.log_manager import LogManager
from preprocessing.converter.impl.csv_converter import CSVConverter
from preprocessing.converter.impl.rds_converter import RDSConverter
from preprocessing.pipeline.abstract_pipeline import AbstractPreprocessingPipeline
from util.file_utils import get_processed_file_path, is_data_file, load_data, save_data


class TabularPreprocessingPipeline(AbstractPreprocessingPipeline):
    """Pipeline for processing tabular data."""

    def __init__(self) -> None:
        """Initialize the tabular preprocessing pipeline."""
        super().__init__(logger=LogManager.get_logger(__name__))
        input_data = self.properties.dataset.input_data
        data_dir = self.properties.system.data_dir

        if not is_data_file(input_data):
            self.logger.error(f"Invalid data file: {input_data}")
            raise ValueError(f"Invalid data file: {input_data}")

        # Get the test data file path based on the training data file path
        test_input_data = input_data.split(".")[0] + "_test." + input_data.split(".")[1]

        # Get input paths for training and test data
        self.train_input_path = f"{data_dir}/{input_data}"
        self.test_input_path = f"{data_dir}/{test_input_data}"

        # Get output paths for processed training and test data
        self.train_output_path = get_processed_file_path(data_dir, input_data, "train")
        self.test_output_path = get_processed_file_path(data_dir, input_data, "test")

    def _execute_pipeline(self) -> None:
        """Execute the preprocessing pipeline for tabular data."""
        self.logger.info("Executing Tabular Preprocessing Pipeline.")

        # Convert and transform the data
        self.__load_and_convert_data(self.train_input_path, self.train_output_path)
        self.__apply_transforms(self.train_output_path)

        # If test data is provided, convert and transform it; otherwise, split the training data
        if os.path.exists(self.test_input_path):
            self.__load_and_convert_data(self.test_input_path, self.test_output_path)
            self.__apply_transforms(self.test_output_path)
        else:
            self.__split_train_test()

    def __load_and_convert_data(self, input_path: str, output_path: str) -> None:
        """Load and convert data if necessary, then save to the processed path."""
        self.logger.info(f"Loading and converting input data: {input_path}")

        input_file_extension = input_path.split(".")[-1]

        if input_file_extension == "csv":
            self.logger.info("Converting CSV to HDF format.")
            csv_converter = CSVConverter()
            csv_converter.convert(input_path, output_path)
        elif input_file_extension == "rds":
            self.logger.info("Converting RDS to HDF format.")
            rds_converter = RDSConverter()
            rds_converter.convert(input_path, output_path)
        else:
            self.logger.error(f"Unsupported file extension: {input_file_extension}")
            raise ValueError(f"Unsupported file extension: {input_file_extension}")

    def __apply_transforms(self, data_path: str) -> None:
        """Apply preprocessing steps to the data if enabled."""
        if self.properties.dataset.enable_transforms:
            self.logger.info("Applying data preprocessing methods")

            # Load data as a torch tensor and apply preprocessing
            data = load_data(data_path)
            data_tensor = torch.tensor(data.values, dtype=torch.float32).to(
                self.properties.system.device
            )

            # Apply each transform to the data
            for transform in self.transforms:
                self.logger.info(f"Applying transform: {transform.__class__.__name__}")
                data_tensor = transform(data_tensor)

            transformed_data = pd.DataFrame(
                data_tensor.cpu().numpy(), columns=data.columns
            )
            save_data(transformed_data, data_path)
            self.logger.info(
                f"Preprocessing steps applied to data and saved to {data_path}"
            )

    def __split_train_test(self) -> None:
        """Split data into training/validation and test sets if test split is defined."""
        self.logger.info("Splitting data into training/validation and test sets")

        # Get properties for data splitting
        test_split = self.properties.dataset.test_split
        seed = self.properties.general.seed

        if test_split <= 0:
            self.logger.info("Test split not required; skipping data splitting.")
            return

        # Load the data
        data = load_data(self.train_output_path)

        # Generate random split indices
        train_val_data, test_data = train_test_split(
            data,
            test_size=test_split,
            random_state=seed,
            shuffle=True,
        )

        self.logger.info(
            f"Splitting dataset: {len(train_val_data)} train/val, {len(test_data)} test"
        )

        # Save the splitted data
        save_data(train_val_data, self.train_output_path)
        save_data(test_data, self.test_output_path)
        self.logger.info(
            f"Data splits saved to {self.train_output_path} and {self.test_output_path}"
        )
