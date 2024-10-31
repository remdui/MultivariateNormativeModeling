"""This module contains the FreeSurferDataset class which is used to load the FreeSurfer dataset."""

from typing import Any

import pandas as pd
from torch.utils.data import Dataset


class FreeSurferDataset(Dataset):
    """This module contains the FreeSurferDataset class which is used to load the FreeSurfer dataset."""

    def __init__(
        self, csv_file: str, covariates_count: int, transform: Any = None
    ) -> None:
        """Constructor for the FreeSurferDataset class."""
        self.data = pd.read_csv(csv_file)
        self.covariates_count = covariates_count
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Returns the brain measures and covariates for a given index."""
        brain_measures = self.data.iloc[
            idx, 0 : -self.covariates_count
        ].values  # Features
        covariates = self.data.iloc[idx, -self.covariates_count :].values  # Covariates

        if self.transform:
            brain_measures = self.transform(brain_measures)
        return brain_measures, covariates

    def get_num_features(self) -> int:
        """Returns the number of features in the dataset."""
        return self.data.shape[1] - self.covariates_count

    def get_num_covariates(self) -> int:
        """Returns the number of covariates in the dataset."""
        return self.covariates_count

    def get_num_samples(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.data.shape[0]
