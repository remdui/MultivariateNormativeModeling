import pandas as pd
from torch.utils.data import Dataset

class FreeSurferDataset(Dataset):
    def __init__(self, csv_file, covariates_count, transform=None):
        self.data = pd.read_csv(csv_file)
        self.covariates_count = covariates_count
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        brain_measures = self.data.iloc[idx, 1:-self.covariates_count].values  # Features
        covariates = self.data.iloc[idx, -self.covariates_count:].values  # Covariates

        if self.transform:
            brain_measures = self.transform(brain_measures)
        return brain_measures, covariates