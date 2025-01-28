"""Data Analysis module to perform data analysis."""

from typing import Any

import pandas as pd
import torch

from analysis.engine.abstract_analysis_engine import AbstractAnalysisEngine
from analysis.metrics.mse import compute_mse
from analysis.metrics.r2 import compute_r2_score
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

    def _get_data_as_tensor(
        self, df: pd.DataFrame, suffix: str, features: list[str]
    ) -> torch.Tensor | None:
        """
        Given a DataFrame `df`, a prefix/suffix like 'orig' or 'recon',.

        and a list of feature names, build a list of columns, extract them,
        and return as a float32 Torch tensor.

        e.g., if suffix='orig', columns become ['orig_feature1', 'orig_feature2', ...].
        """
        # Check if DataFrame is valid
        if df is None or df.empty:
            self.logger.warning("DataFrame is None or empty.")
            return None

        # Build the list of columns
        column_names = [f"{suffix}_{feat}" for feat in features]

        # Check for missing columns
        missing_cols = [c for c in column_names if c not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns in DataFrame: {missing_cols}")
            return None

        # Extract and convert to torch tensor
        values = df[column_names].values  # shape: (num_rows, num_features)
        tensor = torch.as_tensor(values, dtype=torch.float32)
        return tensor

    def calculate_reconstruction_mse(self) -> float:
        """
        Compute MSE between original (input) columns and reconstruction (recon) columns.

        for all features at once, then return it as a float.
        """
        input_tensor = self._get_data_as_tensor(
            self.recon_df, "orig", self.feature_labels
        )
        recon_tensor = self._get_data_as_tensor(
            self.recon_df, "recon", self.feature_labels
        )

        if input_tensor is None or recon_tensor is None:
            self.logger.warning("Unable to calculate MSE due to missing data/tensors.")
            return float("nan")

        # Compute MSE via custom function
        mse_tensor = compute_mse(input_tensor, recon_tensor, metric_type="total")
        mse_value = float(mse_tensor)  # Convert to Python float

        self.logger.info(f"Reconstruction MSE (across all features): {mse_value:.4f}")
        return mse_value

    def calculate_reconstruction_r2(self) -> float:
        """
        Compute R^2 between original (input) columns and reconstruction (recon) columns.

        for all features at once, then return it as a float.
        """
        input_tensor = self._get_data_as_tensor(
            self.recon_df, "orig", self.feature_labels
        )
        recon_tensor = self._get_data_as_tensor(
            self.recon_df, "recon", self.feature_labels
        )

        if input_tensor is None or recon_tensor is None:
            self.logger.warning("Unable to calculate R^2 due to missing data/tensors.")
            return float("nan")

        # Compute R^2 via custom function
        r2_tensor = compute_r2_score(input_tensor, recon_tensor, metric_type="total")
        r2_value = float(r2_tensor)

        self.logger.info(f"Reconstruction R^2 (across all features): {r2_value:.4f}")
        return r2_value
