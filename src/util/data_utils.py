"""Utility functions for data handling."""

import torch
from torch import Tensor
from torch.utils.data import Dataset


def sample_batch_from_indices(dataset: Dataset, indices: list, device: str) -> Tensor:
    """Sample a batch of data from the dataset given the indices.

    Args:
        dataset (Dataset): Dataset to sample from.
        indices (list): List of indices to sample.
        device (torch.device): Device to move the data to.

    Returns:
        Tensor: Batch of data.
    """
    data_slice = [
        dataset[idx][0] for idx in indices
    ]  # Only get the data, not the labels
    data_batch = torch.stack(data_slice).to(device)  # Stack and move to device
    return data_batch
