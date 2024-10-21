from torch.utils.data import DataLoader

from preprocessing.dataset import FreeSurferDataset

class FreeSurferDataloader:
    @staticmethod
    def load_data(properties):
        csv_file = properties.get('dataset')['processed_data']
        data_dir = properties.get('system')['data_dir']
        csv_path = f"{data_dir}/processed/{csv_file}"

        batch_size = properties.get('train')['batch_size']
        shuffle = properties.get('dataset')['shuffle']
        num_workers = properties.get('global')['num_workers']
        covariates_count = properties.get('dataset')['num_covariates']

        # Pass the covariates_count to the dataset
        dataset = FreeSurferDataset(csv_file=csv_path, covariates_count=covariates_count)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader