"""DataLoader for 2D image data."""

from torch.utils.data import DataLoader

from data.abstract_dataloader import AbstractDataloader


class Image2DDataLoader(AbstractDataloader):
    """DataLoader for 2D image data."""

    def train_dataloader(self) -> DataLoader:
        """Get the DataLoader for the training dataset."""
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        """Get the DataLoader for the validation dataset."""
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        """Get the DataLoader for the test dataset."""
        raise NotImplementedError

    def fold_dataloader(self, fold: int) -> tuple[DataLoader, DataLoader]:
        """Get the DataLoader for the training and validation data for a given fold."""
        raise NotImplementedError

    def get_feature_names(self) -> list[str]:
        """Get the names of the features in the dataset."""
        raise NotImplementedError

    def get_covariate_names(self) -> list[str]:
        """Get the names of the covariates in the dataset."""
        raise NotImplementedError

    def get_target_names(self) -> list[str]:
        """Get the names of the targets in the dataset."""
        raise NotImplementedError
