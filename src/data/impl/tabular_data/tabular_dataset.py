"""Tabular implementation of AbstractDataset.

This class loads a tabular dataset from a file, removes specified skipped columns,
and separates feature columns from covariate columns.
"""

import pandas as pd
import torch
from torch import Tensor

from data.abstract_dataset import AbstractDataset
from entities.log_manager import LogManager
from entities.properties import Properties
from util.file_utils import load_data


class TabularDataset(AbstractDataset):
    """
    Dataset class for tabular data.

    Loads data using a file path, removes columns specified as "skipped", and determines
    feature and covariate columns based on configuration properties.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize the TabularDataset.

        Args:
            file_path (str): Path to the file containing the dataset.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)
        self.properties = Properties.get_instance()
        self.logger.debug(f"Loading dataset from: {file_path}")
        self.data = load_data(file_path)

        # Retrieve configuration for skipped and covariate columns.
        self.skipped_columns = self.properties.dataset.skipped_columns or []
        self.all_covariates = self.properties.dataset.covariates
        self.skipped_covariates = self.properties.dataset.skipped_covariates

        if self.skipped_columns:
            self.logger.info(f"Removing skipped columns: {self.skipped_columns}")
            self.skipped_data = self.data[self.skipped_columns].copy()
        else:
            self.skipped_data = pd.DataFrame()

        # Remove skipped columns from main data.
        self.data.drop(columns=self.skipped_columns, inplace=True)

        # Ensure that all specified covariate columns exist.
        missing_covariates = set(self.all_covariates) - set(self.data.columns)
        if missing_covariates:
            raise ValueError(
                f"Covariate columns not found in dataset: {missing_covariates}"
            )

        # Identify feature columns as those not in covariates.
        self.features = [
            col for col in self.data.columns if col not in self.all_covariates
        ]
        self.covariates = [
            col for col in self.all_covariates if col not in self.skipped_covariates
        ]

        self.logger.debug(f"Features: {self.features}")
        self.logger.debug(f"Covariates: {self.covariates}")
        self.logger.debug(f"Skipped Columns: {self.skipped_columns}")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of rows in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Retrieve the features and covariates for the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor]: A tuple where the first element is the features tensor
            (shape [num_features]) and the second is the covariates tensor (shape [num_covariates]).
        """
        row = self.data.iloc[idx]
        # Extract features and covariates based on column names.
        features_series: pd.Series = row[self.features]
        covariates_series: pd.Series = row[self.covariates]
        # Convert to torch.float32 tensors.
        features = torch.tensor(features_series.to_numpy(), dtype=torch.float32)
        covariates = torch.tensor(covariates_series.to_numpy(), dtype=torch.float32)
        return features, covariates

    def get_num_features(self) -> int:
        """
        Return the number of feature columns in the dataset.

        Returns:
            int: Number of features (excluding covariates).
        """
        return len(self.features)

    def get_num_covariates(self) -> int:
        """
        Return the number of covariate columns in the dataset.

        Returns:
            int: Number of covariates.
        """
        return len(self.covariates)

    def get_skipped_data(self) -> pd.DataFrame:
        """
        Return a DataFrame containing the skipped columns.

        Returns:
            pd.DataFrame: DataFrame of skipped columns.
        """
        return self.skipped_data
