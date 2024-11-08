"""Utility functions for PyTorch tensors."""

import torch
from torch import Tensor


def shuffle_tensor(data: Tensor) -> Tensor:
    """Shuffle a tensor along dimension 0.

    Args:
        data (Tensor): Input tensor to shuffle.

    Returns:
        Tensor: Shuffled tensor.
    """
    # Generate a random permutation of indices along dim 0
    indices = torch.randperm(data.size(0), device=data.device)
    # Reorder data along dim 0 using the random indices
    shuffled_data = data[indices]
    return shuffled_data
