from torch.utils.data import DataLoader

from entities.properties import Properties
from preprocessing.dataset import FreeSurferDataset

class FreeSurferDataloader:
    @staticmethod
    def init_dataloader():
        properties = Properties.get_instance()

        csv_file = properties.dataset.processed_data_file
        data_dir = properties.system.data_dir
        csv_path = f"{data_dir}/processed/{csv_file}"

        batch_size = properties.train.batch_size
        shuffle = properties.dataset.shuffle
        num_workers = properties.system.num_workers
        covariates_count = properties.dataset.num_covariates

        # Pass the covariates_count to the dataset
        dataset = FreeSurferDataset(csv_file=csv_path, covariates_count=covariates_count)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader