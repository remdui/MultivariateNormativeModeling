# preprocessing/preprocessing_pipeline.py

"""Defines the PreprocessingPipeline class."""

import pandas as pd

from data.convertion.abstract_dataconverter import AbstractDataConverter
from data.convertion.r_to_csv_converter import RDSToCSVDataConverter
from data.preprocessing.abstract_preprocessor import AbstractPreprocessor
from data.preprocessing.impl.data_cleaning import DataCleaningPreprocessor
from data.preprocessing.impl.normalization import NormalizationPreprocessor
from entities.log_manager import LogManager
from entities.properties import Properties

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

        # instantiate the data converter
        self.__init_data_converter()

        # instantiate the preprocessors
        self.__init_preprocessors()

    def __init_data_converter(self) -> None:
        """Initialize the data converter."""
        converter_name = self.properties.dataset.data_converter

        if converter_name in DATA_CONVERTER_MAPPING:
            converter_class = DATA_CONVERTER_MAPPING[converter_name]
            self.data_converter = converter_class()
            self.logger.info(f"Using data converter: {converter_name}")
        elif converter_name == "":
            self.logger.info("No data converter specified in configuration")
        else:
            self.logger.error(f"Unknown data converter: {converter_name}")
            raise ValueError(f"Unknown data converter: {converter_name}")

    def __init_preprocessors(self) -> None:
        """Initialize the preprocessors."""
        self.preprocessors = []
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
        raw_data = self.properties.dataset.raw_data
        input_data = self.properties.dataset.input_data

        # Step 1: Data Conversion
        if hasattr(self, "data_converter") and self.data_converter:
            self.logger.info("Starting data conversion")
            self.data_converter.convert(
                input_file_name=raw_data, output_file_name=input_data
            )

        # Step 2: Preprocessing Steps
        if self.properties.dataset.enable_preprocessing:
            processed_data_file = (
                self.properties.system.data_dir + "/processed/" + input_data
            )
            self.logger.info("Loading converted data for further processing")
            data = pd.read_csv(processed_data_file)

            for preprocessor in self.preprocessors:
                self.logger.info(
                    f"Applying preprocessor: {preprocessor.__class__.__name__}"
                )
                data = preprocessor.process(data)
                self.logger.info(f"Saving processed data to {processed_data_file}")
                data.to_csv(processed_data_file, index=False)
                self.logger.info("Preprocessing pipeline completed successfully")
