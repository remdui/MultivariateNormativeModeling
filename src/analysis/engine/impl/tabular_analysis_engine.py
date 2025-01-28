"""Data Analysis module to perform data analysis."""

from typing import Any

import pandas as pd

from analysis.engine.abstract_analysis_engine import AbstractAnalysisEngine
from entities.log_manager import LogManager
from util.file_utils import (
    get_internal_file_extension,
    get_processed_file_path,
    is_data_file,
    load_data,
)


class TabularAnalysisEngine(AbstractAnalysisEngine):
    """Class to perform data analysis for tabular data."""

    def __init__(self) -> None:
        """Initialize the DataAnalysis object."""
        super().__init__(LogManager.get_logger(__name__))

        # Define data types
        self.covariate_labels: list[str] = []
        self.feature_labels: list[str] = []
        self.target_labels: list[str] = []

        # Internal references to loaded DataFrames
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.recon_df: pd.DataFrame = pd.DataFrame()

    def initialize_engine(self, *args: Any, **kwargs: Any) -> None:
        """Initialize tabular exploration pipeline."""
        self.logger.info("Initializing Tabular Data Exploration.")

        # Store references for later usage
        self.feature_labels = kwargs.get("feature_labels", [])
        self.covariate_labels = kwargs.get("covariate_labels", [])
        self.target_labels = kwargs.get("target_labels", [])

        # ---------------------------------------------------------
        # 1) Load the processed train/test data
        # ---------------------------------------------------------
        data_dir = self.properties.system.data_dir
        input_data = self.properties.dataset.input_data

        if not is_data_file(input_data):
            self.logger.error(f"Invalid data file: {input_data}")
            raise ValueError(f"Invalid data file: {input_data}")

        # Processed data file paths
        train_output_path = get_processed_file_path(data_dir, input_data, "train")
        test_output_path = get_processed_file_path(data_dir, input_data, "test")

        self.logger.info("Loading processed training/test data...")
        # Load the processed data
        self.train_df = load_data(train_output_path)  # e.g. DF with columns
        self.test_df = load_data(test_output_path)  # e.g. DF with columns
        self.logger.info(f"Loaded train_df: {self.train_df.shape}")
        self.logger.info(f"Loaded test_df: {self.test_df.shape}")

        # ---------------------------------------------------------
        # 2) Load the new reconstruction data created by ValidateTask
        # ---------------------------------------------------------
        output_extension = get_internal_file_extension()
        recon_file_path = (
            f"{self.properties.system.output_dir}/reconstructions/"
            f"{self.properties.model_name}_validation_data.{output_extension}"
        )

        # If you want to confirm existence before loading, do so here
        self.logger.info(f"Loading reconstruction data from {recon_file_path}...")
        self.recon_df = load_data(recon_file_path)

        self.logger.info(f"Loaded recon_df: {self.recon_df.shape}")

        self.logger.info("Tabular Data Exploration engine initialization complete.")

    def calculate_reconstruction_mse(self) -> float:
        return 0.0
