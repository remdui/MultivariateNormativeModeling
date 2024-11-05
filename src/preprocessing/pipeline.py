"""Defines the PreprocessingPipeline class."""

import pandas as pd

from entities.log_manager import LogManager
from entities.properties import Properties
from preprocessing.abstract_preprocessor import AbstractPreprocessor
from preprocessing.converter.abstract_dataconverter import AbstractDataConverter
from preprocessing.converter.impl.r_to_csv_converter import RDSToCSVDataConverter
from preprocessing.impl.data_cleaning import DataCleaningPreprocessor
from preprocessing.impl.normalization import NormalizationPreprocessor
from util.file_utils import get_processed_file_path, is_data_file

DATA_CONVERTER_MAPPING: dict[str, type[AbstractDataConverter]] = {
    "RdsToCsvConverter": RDSToCSVDataConverter,
}

PREPROCESSOR_MAPPING: dict[str, type[AbstractPreprocessor]] = {
    "NormalizationPreprocessor": NormalizationPreprocessor,
    "DataCleaningPreprocessor": DataCleaningPreprocessor,
}


class PreprocessingPipeline:
    """Class to manage data conversion and preprocessing steps."""

    def __init__(self) -> None:
        """Initialize the preprocessing pipeline.

        Args:
            data_converter (Type[AbstractDataConverter]): Data converter class to use.
            preprocessors (Optional[List[Type[AbstractPreprocessor]]]): List of preprocessor classes to apply.
        """
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()
        self.preprocessors: list[AbstractPreprocessor] = []

        # instantiate the preprocessors
        self.__init_preprocessors()

    def __init_preprocessors(self) -> None:
        """Initialize the preprocessors."""
        for preprocessor_config in self.properties.dataset.preprocessors:
            preprocessor_name = preprocessor_config.name
            preprocessor_params = preprocessor_config.params or {}
            if preprocessor_name in PREPROCESSOR_MAPPING:
                preprocessor_class = PREPROCESSOR_MAPPING[preprocessor_name]
                preprocessor_instance = preprocessor_class(**preprocessor_params)
                self.preprocessors.append(preprocessor_instance)
                self.logger.info(f"Added preprocessor: {preprocessor_name}")
            else:
                self.logger.error(f"Unknown preprocessor: {preprocessor_name}")
                raise ValueError(f"Unknown preprocessor: {preprocessor_name}")

    def run(self) -> None:
        """Run the data conversion and preprocessing pipeline."""
        # Paths from properties
        input_data = self.properties.dataset.input_data
        data_dir = self.properties.system.data_dir
        internal_data_type = self.properties.dataset.data_type

        if internal_data_type == "tabular":
            self.__execute_tabular_pipeline(input_data, data_dir)
        if internal_data_type in {"image2d", "image3d"}:
            self.__execute_image_pipeline(input_data, data_dir)

        self.logger.info("Preprocessing pipeline completed")

    def __execute_tabular_pipeline(self, input_data: str, data_dir: str) -> None:
        """Execute the tabular data pipeline."""
        # Check if the input data is a file, then proceed with the pipeline for tabular data
        if is_data_file(input_data):
            # Get the file name and extension
            input_file_name, input_file_extension = input_data.split(".")

            # Get the processed data file path
            processed_data_file = get_processed_file_path(data_dir, input_data)

            # Step 1: Data Conversion
            if input_file_extension == "csv":
                self.logger.info(
                    f"Data is already in CSV format, no conversion needed. Saving to {processed_data_file}"
                )
                data = pd.read_csv(f"{data_dir}/{input_data}")
                data.to_csv(processed_data_file, index=False)

            if input_file_extension == "rds":
                self.logger.info(
                    f"Data is in RDS format. Saving to{processed_data_file}"
                )
                data_converter = RDSToCSVDataConverter()
                csv_file_name = input_file_name + ".csv"
                data_converter.convert(input_data, csv_file_name)

            # Step 2: Preprocessing Steps
            if self.properties.dataset.enable_preprocessing:
                self.logger.info("Loading data for further processing")
                data = pd.read_csv(processed_data_file)

                for preprocessor in self.preprocessors:
                    self.logger.info(
                        f"Applying preprocessor: {preprocessor.__class__.__name__}"
                    )
                    data = preprocessor.process(data)
                    self.logger.info(f"Saving processed data to {processed_data_file}")
                    data.to_csv(processed_data_file, index=False)

            # Step 3: Data Splitting
            test_split = self.properties.dataset.test_split

            # If test_split is set to 0, only train and validation splits are created
            if test_split > 0:
                self.logger.info("Splitting data into train/validation, and test sets")
                train_val_split = 1 - test_split
                dataset = pd.read_csv(processed_data_file)
                dataset_size = len(dataset)

                # Calculate split sizes
                train_val_size = int(train_val_split * dataset_size)
                test_size = int(test_split * dataset_size)

                self.logger.info(
                    f"Splitting dataset: {train_val_size} train/val, {test_size} test"
                )

                # Split the csv file into train/val and test sets
                train_val_data = dataset.iloc[:train_val_size]
                test_data = dataset.iloc[train_val_size:]

                # Save the train/val and test data
                train_val_data.to_csv(
                    processed_data_file,
                    index=False,
                )

                test_data.to_csv(
                    get_processed_file_path(data_dir, "test_" + input_data),
                    index=False,
                )

    def __execute_image_pipeline(self, input_data: str, data_dir: str) -> None:
        """Execute the image data pipeline."""
        raise NotImplementedError
