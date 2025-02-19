"""Implementation of the Tabular Data Preprocessing Pipeline.

This module defines the TabularPreprocessingPipeline, which handles conversion,
transformation, and splitting of tabular datasets based on configuration from the
Properties instance. It supports converting raw files (CSV, RDS) to HDF format, applying
a sequence of preprocessing transforms, and splitting the data into training/validation and
test sets with optional grouping to avoid data leakage.
"""

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
    """
    Pipeline for processing tabular data.

    This pipeline loads raw data, converts it to a processed format, applies a sequence
    of preprocessing transforms, and splits the data into training/validation and test sets.
    The configuration (input file, transforms, splitting parameters) is obtained from the
    Properties instance.
    """

    def __init__(self) -> None:
        """
        Initialize the TabularPreprocessingPipeline.

        Verifies the input file, sets up input and output paths for training and test data,
        and prepares the pipeline configuration.

        Raises:
            ValueError: If the provided data file is not valid.
        """
        super().__init__(logger=LogManager.get_logger(__name__))
        input_data = self.properties.dataset.input_data
        data_dir = self.properties.system.data_dir

        if not is_data_file(input_data):
            self.logger.error(f"Invalid data file: {input_data}")
            raise ValueError(f"Invalid data file: {input_data}")

        # Derive test file name by appending '_test' to the base filename.
        base_name, ext = input_data.split(".", 1)
        test_input_data = f"{base_name}_test.{ext}"

        # Set full input paths.
        self.train_input_path = os.path.join(data_dir, input_data)
        self.test_input_path = os.path.join(data_dir, test_input_data)

        # Set full output paths for processed data.
        self.train_output_path = get_processed_file_path(data_dir, input_data, "train")
        self.test_output_path = get_processed_file_path(data_dir, input_data, "test")

    def _execute_pipeline(self) -> None:
        """
        Execute the tabular preprocessing pipeline.

        The pipeline converts and transforms the training data. If a test data file exists,
        it is processed similarly; otherwise, the training data is split into train/validation
        and test sets.
        """
        self.logger.info("Executing Tabular Preprocessing Pipeline.")

        # Process training data.
        self.__load_and_convert_data(str(self.train_input_path), self.train_output_path)
        self.__apply_transforms(self.train_output_path)

        # Process test data if available; otherwise, perform train/test split.
        if os.path.exists(self.test_input_path):
            self.__load_and_convert_data(self.test_input_path, self.test_output_path)
            self.__apply_transforms(self.test_output_path)
        else:
            self.__split_train_test()

    def __load_and_convert_data(self, input_path: str, output_path: str) -> None:
        """
        Load raw data, convert it to HDF format, and save to the processed path.

        The conversion is based on the file extension (CSV or RDS).

        Args:
            input_path (str): Path to the raw input data.
            output_path (str): Destination path for the processed data.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        self.logger.info(f"Loading and converting input data: {input_path}")
        input_file_extension = input_path.split(".")[-1]

        if input_file_extension == "csv":
            self.logger.info("Converting CSV to HDF format.")
            converter = CSVConverter()
            converter.convert(input_path, output_path)
        elif input_file_extension == "rds":
            self.logger.info("Converting RDS to HDF format.")
            converter = RDSConverter()
            converter.convert(input_path, output_path)
        else:
            self.logger.error(f"Unsupported file extension: {input_file_extension}")
            raise ValueError(f"Unsupported file extension: {input_file_extension}")

    def __apply_transforms(self, data_path: str) -> None:
        """
        Apply preprocessing transforms to the processed data.

        This method loads data from the given path, separates numeric columns from columns
        to be skipped, applies each transform sequentially (dropping rows with NaN values after
        each transform), and then merges the skipped columns back before saving the cleaned data.

        Args:
            data_path (str): Path to the processed data file.
        """
        if not self.properties.dataset.enable_transforms:
            return

        self.logger.info("Applying data preprocessing methods")
        # 1. Load the processed data.
        data = load_data(data_path)

        # 2. Separate skipped columns and numeric columns.
        skipped_columns = self.properties.dataset.skipped_columns
        skipped_data = data[skipped_columns].copy()
        numeric_data = data.drop(columns=skipped_columns)

        # 3. Preserve original indices for alignment.
        numeric_indices = numeric_data.index

        # 4. Convert numeric data to a tensor.
        device = self.properties.system.device
        data_tensor = torch.tensor(
            numeric_data.values, dtype=torch.float32, device=device
        )

        # 5. Apply each transform and drop rows with NaN values immediately.
        for transform in self.transforms:
            self.logger.info(f"Applying transform: {transform.__class__.__name__}")
            data_tensor = transform(data_tensor)
            nan_mask = torch.isnan(data_tensor).any(dim=1)
            num_bad_rows = nan_mask.sum().item()

            if num_bad_rows > 0:
                self.logger.warning(
                    f"Found {num_bad_rows} rows with NaN values after {transform.__class__.__name__}. Dropping these rows."
                )
                # Identify indices of rows with NaNs.
                bad_index_positions = nan_mask.nonzero(as_tuple=True)[0]
                bad_indices = numeric_indices[bad_index_positions.cpu().numpy()]

                # Remove rows with NaNs.
                keep_mask = ~nan_mask
                data_tensor = data_tensor[keep_mask]
                numeric_indices = numeric_indices[keep_mask.cpu().numpy()]
                skipped_data.drop(index=bad_indices, inplace=True)

        # 6. Convert the cleaned tensor back to a DataFrame.
        transformed_numeric_data = pd.DataFrame(
            data_tensor.cpu().numpy(),
            index=numeric_indices,
            columns=numeric_data.columns,
        )

        # 7. Merge skipped columns back in, aligning by index.
        transformed_data = transformed_numeric_data.join(skipped_data, how="left")
        expected_order = list(skipped_data.columns) + list(
            transformed_numeric_data.columns
        )
        transformed_data = transformed_data[expected_order]

        # 8. Save the final cleaned data.
        save_data(transformed_data, data_path)
        self.logger.info(
            f"Preprocessing steps applied to data and saved to {data_path}"
        )

    def __split_train_test(self) -> None:
        """
        Split the processed data into training/validation and test sets without data leakage.

        If grouping columns are specified, the data is grouped to avoid leakage across groups.
        Otherwise, a random split is performed.
        """
        self.logger.info("Splitting data into training/validation and test sets")
        test_split = self.properties.dataset.test_split
        seed = self.properties.general.seed
        group_cols = self.properties.dataset.row_data_leakage_columns

        if test_split <= 0:
            self.logger.info("Test split not required; skipping data splitting.")
            return

        # Load processed training data.
        data = load_data(self.train_output_path)

        if group_cols:
            self.logger.info(f"Grouping data on columns to avoid leakage: {group_cols}")
            # Create a group identifier by combining specified columns.
            data["__group_id__"] = data[group_cols].astype(str).agg("-".join, axis=1)
            unique_groups = data["__group_id__"].unique()
            n_test_groups = int(len(unique_groups) * test_split)
            # Shuffle groups for reproducibility.
            unique_groups = (
                pd.Series(unique_groups).sample(frac=1, random_state=seed).values
            )
            test_groups = set(unique_groups[:n_test_groups])
            train_val_groups = set(unique_groups[n_test_groups:])
            # Log any group overlap (should not occur ideally).
            overlap = train_val_groups.intersection(test_groups)
            if overlap:
                self.logger.warning(
                    f"Overlap detected in group IDs between train/val and test sets: {overlap}"
                )
            else:
                self.logger.info(
                    "No overlap detected in group IDs between train/val and test sets."
                )
            train_val_data = data[data["__group_id__"].isin(train_val_groups)].drop(
                "__group_id__", axis=1
            )
            test_data = data[data["__group_id__"].isin(test_groups)].drop(
                "__group_id__", axis=1
            )
        else:
            train_val_data, test_data = train_test_split(
                data, test_size=test_split, random_state=seed, shuffle=True
            )

        self.logger.info(
            f"Dataset split: {len(train_val_data)} train/val, {len(test_data)} test"
        )
        save_data(train_val_data, self.train_output_path)
        save_data(test_data, self.test_output_path)
        self.logger.info(
            f"Data splits saved to {self.train_output_path} and {self.test_output_path}"
        )
