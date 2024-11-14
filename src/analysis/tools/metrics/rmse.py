"""Root Mean Squared Error (RMSE) computation for tensors."""

import torch
from torch import Tensor


def compute_rmse(
    original: Tensor,
    reconstructed: Tensor,
    rmse_type: str = "total",
) -> Tensor:
    """Compute Root Mean Squared Error (RMSE) for different types.

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        rmse_type (str): Type of RMSE to compute. Options are 'sample', 'feature', 'total'.

    Returns:
        Tensor: Computed RMSE based on the specified type.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Calculate squared error
    squared_error = (original - reconstructed) ** 2

    # RMSE per sample
    if rmse_type == "sample":
        # Mean squared error over features for each sample, then square root
        rmse = torch.sqrt(squared_error.mean(dim=1))

    # RMSE per feature
    elif rmse_type == "feature":
        # Mean squared error over samples for each feature, then square root
        rmse = torch.sqrt(squared_error.mean(dim=0))

    # RMSE total
    elif rmse_type == "total":
        # Mean squared error over all samples and features, then square root
        rmse = torch.sqrt(squared_error.mean())

    else:
        raise ValueError(
            f"Invalid rmse_type: {rmse_type}. Choose from 'sample', 'feature', 'total'."
        )

    return rmse
