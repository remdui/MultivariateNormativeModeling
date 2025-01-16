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

        # Remove skipped columns from the processed data
        self.__remove_skipped_columns()

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
        """Apply preprocessing steps to the data if enabled,.

        dropping rows with NaN after each transform (and removing
        from the 'skipped' columns as well).
        """
        if not self.properties.dataset.enable_transforms:
            return

        self.logger.info("Applying data preprocessing methods")

        # 1. Load data
        data = load_data(data_path)

        # 2. Separate out skipped columns and numeric columns
        skipped_columns = self.properties.dataset.skipped_columns
        skipped_data = data[skipped_columns].copy()
        numeric_data = data.drop(columns=skipped_columns)

        # 3. Store the original indices so we can align numeric_data and skipped_data
        numeric_indices = numeric_data.index

        # 4. Convert numeric data to torch.Tensor
        device = self.properties.system.device
        data_tensor = torch.tensor(
            numeric_data.values, dtype=torch.float32, device=device
        )

        # 5. Apply each transform, then immediately drop NaN rows
        for transform in self.transforms:
            self.logger.info(f"Applying transform: {transform.__class__.__name__}")

            # Apply the transform in-place
            data_tensor = transform(data_tensor)

            # 5a. Check if any row now has a NaN (in *any* column)
            #     'mask' is True for rows with at least one NaN
            nan_mask = torch.isnan(data_tensor).any(dim=1)
            num_bad_rows = nan_mask.sum().item()

            if num_bad_rows > 0:
                self.logger.warning(
                    f"Found {num_bad_rows} rows with NaN values after "
                    f"{transform.__class__.__name__}. Dropping these rows."
                )

                # 5b. Figure out which row indices correspond to these NaNs
                bad_index_positions = nan_mask.nonzero(as_tuple=True)[
                    0
                ]  # Tensor indices
                bad_indices = numeric_indices[bad_index_positions.cpu().numpy()]

                # 5c. Remove those rows from 'data_tensor'
                keep_mask = ~nan_mask
                data_tensor = data_tensor[keep_mask]

                # 5d. Remove those rows from 'numeric_indices'
                numeric_indices = numeric_indices[keep_mask.cpu().numpy()]

                # 5e. Also remove them from 'skipped_data'
                skipped_data.drop(index=bad_indices, inplace=True)

        # 6. At this point, data_tensor and numeric_indices only contain valid rows
        #    (i.e., no NaNs across transforms).
        #    Convert back to a DataFrame using the updated indices.
        transformed_numeric_data = pd.DataFrame(
            data_tensor.cpu().numpy(),
            index=numeric_indices,
            columns=numeric_data.columns,
        )

        # 7. Merge the skipped columns back in, aligning on the index
        transformed_data = transformed_numeric_data.join(skipped_data, how="left")

        # 8. Save the final, cleaned DataFrame
        save_data(transformed_data, data_path)
        self.logger.info(
            f"Preprocessing steps applied to data and saved to {data_path}"
        )

    def __split_train_test(self) -> None:
        """Split data into training/validation and test sets without data leakage.

        across groups defined by row_data_leakage_columns.
        """
        self.logger.info("Splitting data into training/validation and test sets")

        test_split = self.properties.dataset.test_split
        seed = self.properties.general.seed
        group_cols = self.properties.dataset.row_data_leakage_columns

        if test_split <= 0:
            self.logger.info("Test split not required; skipping data splitting.")
            return

        # Load the processed training data
        data = load_data(self.train_output_path)

        if group_cols:
            self.logger.info(
                f"Ensuring no data leakage by grouping on columns: {group_cols}"
            )

            # Create a single group identifier by combining the specified group columns
            data["__group_id__"] = data[group_cols].astype(str).agg("-".join, axis=1)

            # Extract unique groups
            unique_groups = data["__group_id__"].unique()

            # Determine the number of groups to place in the test set
            n_test_groups = int(len(unique_groups) * test_split)

            # Shuffle groups with a fixed seed for reproducibility
            unique_groups = (
                pd.Series(unique_groups).sample(frac=1, random_state=seed).values
            )

            # Split groups into test and train/val sets
            test_groups = set(unique_groups[:n_test_groups])
            train_val_groups = set(unique_groups[n_test_groups:])

            # Verify no overlap of groups between train/val and test
            overlap = train_val_groups.intersection(test_groups)
            if overlap:
                self.logger.warning(
                    f"Overlap detected in group IDs between train/val and test sets: {overlap}"
                )
            else:
                self.logger.info(
                    "No overlap detected in group IDs between train/val and test sets."
                )

            # Split data based on groups
            train_val_data = data[data["__group_id__"].isin(train_val_groups)].drop(
                "__group_id__", axis=1
            )
            test_data = data[data["__group_id__"].isin(test_groups)].drop(
                "__group_id__", axis=1
            )
        else:
            # If no grouping column specified, do a normal split
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

    def __remove_skipped_columns(self) -> None:
        """Remove skipped columns from the processed data."""
        skipped_columns = self.properties.dataset.skipped_columns
        if skipped_columns:
            self.logger.info(f"Removing skipped columns: {skipped_columns}")

            # Load the processed training data and test data
            train_data = load_data(self.train_output_path)
            test_data = load_data(self.test_output_path)

            # Remove skipped columns
            train_data.drop(columns=skipped_columns, inplace=True)
            test_data.drop(columns=skipped_columns, inplace=True)

            # Save the modified data
            save_data(train_data, self.train_output_path)
            save_data(test_data, self.test_output_path)
            self.logger.info(
                f"Skipped columns removed from data and saved to {self.train_output_path} and {self.test_output_path}"
            )
