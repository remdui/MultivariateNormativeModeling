"""Tabular implementation of AbstractDataset.

This class loads a tabular dataset from a file, removes specified skipped columns,
and separates feature columns from covariate columns.
"""

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Subset

from data.abstract_dataset import AbstractDataset
from entities.log_manager import LogManager
from entities.properties import Properties
from util.file_utils import load_data


class TabularSubset(Subset):
    """Custom Subset class to retain TabularDataset features."""

    def get_skipped_data(self) -> pd.DataFrame:
        """
        Return the skipped data for the rows in this subset by subsetting the DataFrame.

        from the original dataset.
        """
        df = self.dataset.get_skipped_data()
        # Use the stored indices to slice the skipped data accordingly.
        return df.iloc[self.indices]


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
        self.covariates = self.properties.dataset.covariates
        self.skipped_covariates = self.properties.dataset.skipped_covariates

        if self.skipped_columns:
            self.logger.info(f"Removing skipped columns: {self.skipped_columns}")
            self.skipped_data = self.data[self.skipped_columns].copy()
        else:
            self.skipped_data = pd.DataFrame()

        # Remove skipped columns from main data.
        self.data.drop(columns=self.skipped_columns, inplace=True)

        # Handle missing covariates due to one-hot encoding
        self.covariates = self._adjust_for_one_hot_encoding(self.covariates)
        self.skipped_covariates = self._adjust_for_one_hot_encoding(
            self.skipped_covariates
        )

        # Identify feature columns as those not in covariates.
        self.features = [col for col in self.data.columns if col not in self.covariates]
        self.covariates = [
            col for col in self.covariates if col not in self.skipped_covariates
        ]

        self.logger.debug(f"Features: {self.features}")
        self.logger.debug(f"Covariates: {self.covariates}")
        self.logger.debug(f"Skipped Columns: {self.skipped_columns}")

    def _adjust_for_one_hot_encoding(self, features: list) -> list:
        """
        Adjust feature names to account for one-hot encoding.

        If a feature is missing from `self.data.columns`, check if it exists as one-hot encoded
        columns (prefix-based match). If found, replace the original feature name with its
        corresponding one-hot encoded feature names.

        Returns:
            List[str]: Updated list of features.
        """
        updated_features = []
        for feature in features:
            if feature in self.data.columns:
                # exists as-is, keep it
                updated_features.append(feature)
            else:
                # Check for one-hot encoded versions
                one_hot_features = [
                    col for col in self.data.columns if col.startswith(f"{feature}_")
                ]
                if one_hot_features:
                    self.logger.info(
                        f"Replacing feature '{feature}' with one-hot encoded features: {one_hot_features}"
                    )
                    updated_features.extend(one_hot_features)
                else:
                    self.logger.warning(
                        f"Feature '{feature}' not found in dataset and no one-hot encoded features detected."
                    )

        return updated_features

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
        features_series: pd.Series = row[self.features]
        covariates_series: pd.Series = row[self.covariates]
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
