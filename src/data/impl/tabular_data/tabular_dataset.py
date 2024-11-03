"""This module contains the FreeSurferDataset class which is used to load the FreeSurfer dataset."""

import pandas as pd
import torch

from data.abstract_dataset import AbstractDataset


class TabularDataset(AbstractDataset):
    """Dataset class to load the Tabular dataset."""

    def __init__(self, csv_file: str, covariates_count: int) -> None:
        """Constructor for the TabularDataset class.

        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            covariates_count (int): Number of covariate columns at the end of each row.
            transform (Optional[Any], optional): Transformations to apply to the data. Defaults to None.
        """
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.covariates_count = covariates_count

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
                - covariates (torch.Tensor): The covariates tensor of shape [covariates_count].
        """
        # Extract brain measures and covariates from the DataFrame
        features_df = self.data.iloc[idx, 0 : -self.covariates_count].values  # Features

        covariates_df = self.data.iloc[
            idx, -self.covariates_count :
        ].values  # Covariates

        # Convert to torch.float32 tensors
        features = torch.tensor(features_df, dtype=torch.float32)
        covariates = torch.tensor(covariates_df, dtype=torch.float32)

        return features, covariates

    def get_num_features(self) -> int:
        """Returns the number of features in the dataset.

        Returns:
            int: Number of features (excluding covariates).
        """
        return self.data.shape[1] - self.covariates_count

    def get_num_covariates(self) -> int:
        """Returns the number of covariates in the dataset.

        Returns:
            int: Number of covariates.
        """
        return self.covariates_count
