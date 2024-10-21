from torch.utils.data import DataLoader

from src.preprocessing.dataset import FreeSurferDataset
from src.util.config_utils import ConfigLoader


class FreeSurferDataloader:
    @staticmethod
    def load_data(csv_path, config_file, batch_size=32, shuffle=True, num_workers=4):
        # Create an instance of ConfigLoader and load the configuration
        config_loader = ConfigLoader(config_file)
        config = config_loader.load_config()

        covariates_count = config['dataset']['covariates_count']

        # Pass the covariates_count to the dataset
        dataset = FreeSurferDataset(csv_file=csv_path, covariates_count=covariates_count)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader