"""DataLoader for 3D image data."""

from torch.utils.data import DataLoader

from data.abstract_dataloader import AbstractDataloader


class Image3DDataLoader(AbstractDataloader):
    """DataLoader for 3D image data."""

    def train_dataloader(self) -> DataLoader:
        """Get the DataLoader for the training dataset."""
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        """Get the DataLoader for the validation dataset."""
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        """Get the DataLoader for the test dataset."""
        raise NotImplementedError
