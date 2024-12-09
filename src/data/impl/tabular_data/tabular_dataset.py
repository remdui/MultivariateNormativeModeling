"""TabularDataset class to process Tabular datasets."""

import pandas as pd
import torch

from data.abstract_dataset import AbstractDataset
from entities.log_manager import LogManager
from util.file_utils import load_data


class TabularDataset(AbstractDataset):
    """Dataset class to load the Tabular dataset."""

    def __init__(self, file_path: str, covariates: list[str]) -> None:
        """Constructor for the TabularDataset class.

        Args:
            file_path (str): Path to the file containing the dataset.
            covariatess (list[str]): List of covariates.
        """
        super().__init__()
        self.logger = LogManager.get_logger(__name__)

        self.logger.debug(f"Loading dataset from: {file_path}")
        self.data = load_data(file_path)
        self.covariates = covariates

        # Ensure covariate names exist in the dataset
        missing_covariates = set(self.covariates) - set(self.data.columns)
        if missing_covariates:
            raise ValueError(
                f"Covariate columns not found in dataset: {missing_covariates}"
            )

        # Identify feature columns (excluding covariates)
        self.features = [col for col in self.data.columns if col not in covariates]

        # If debug mode is enabled, log the dataset details
        self.logger.debug(f"Features: {self.features}")
        self.logger.debug(f"Covariates: {self.covariates}")

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Returns the features and covariates for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - features (torch.Tensor): The input features tensor of shape [num_features].
                - covariates (torch.Tensor): The covariates tensor of shape [num_covariates].
        """
        # Get the row as a Pandas Series using .iloc
        row = self.data.iloc[idx]

        # Extract features and covariates based on column names
        features_series: pd.Series = row[self.features]
        covariates_series: pd.Series = row[self.covariates]

        # Convert to torch.float32 tensors
        features = torch.tensor(features_series.to_numpy(), dtype=torch.float32)
        covariates = torch.tensor(covariates_series.to_numpy(), dtype=torch.float32)

        return features, covariates

    def get_num_features(self) -> int:
        """Returns the number of features in the dataset.

        Returns:
            int: Number of features (excluding covariates).
        """
        return len(self.features)

    def get_num_covariates(self) -> int:
        """Returns the number of covariates in the dataset.

        Returns:
            int: Number of covariates.
        """
        return len(self.covariates)
