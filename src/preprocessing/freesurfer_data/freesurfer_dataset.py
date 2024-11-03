"""This module contains the FreeSurferDataset class which is used to load the FreeSurfer dataset."""

from typing import Any

import pandas as pd
import torch

from preprocessing.abstract_dataset import AbstractDataset


class FreeSurferDataset(AbstractDataset):
    """Dataset class to load the FreeSurfer dataset."""

    def __init__(
        self, csv_file: str, covariates_count: int, transform: Any = None
    ) -> None:
        """Constructor for the FreeSurferDataset class.

        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            covariates_count (int): Number of covariate columns at the end of each row.
            transform (Optional[Any], optional): Transformations to apply to the data. Defaults to None.
        """
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.covariates_count = covariates_count
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Returns the brain measures and covariates for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - brain_measures (torch.Tensor): The input features tensor of shape [num_features].
                - covariates (torch.Tensor): The covariates tensor of shape [covariates_count].
        """
        # Extract brain measures and covariates from the DataFrame
        brain_measures_df = self.data.iloc[
            idx, 0 : -self.covariates_count
        ].values  # Features

        covariates_df = self.data.iloc[
            idx, -self.covariates_count :
        ].values  # Covariates

        # Convert to torch.float32 tensors
        brain_measures = torch.tensor(brain_measures_df, dtype=torch.float32)
        covariates = torch.tensor(covariates_df, dtype=torch.float32)

        # Apply transformations if any
        if self.transform:
            brain_measures = self.transform(brain_measures)

        return brain_measures, covariates

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
