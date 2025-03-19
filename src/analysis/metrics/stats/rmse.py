"""Root Mean Squared Error (RMSE) computation for tensors."""

import torch
from torch import Tensor


def compute_rmse(
    original: Tensor,
    reconstructed: Tensor,
    metric_type: str = "total",
) -> Tensor:
    """Compute Root Mean Squared Error (RMSE) for different types.

    MSE provides the square root of the average squared differences, combining sensitivity to large errors with interpretability on the same scale as the original data

    Equation: RMSE = sqrt(1/n * Σ (original - reconstructed)²)

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        metric_type (str): Type of RMSE to compute. Options are 'sample', 'feature', 'total'.

    Returns:
        Tensor: Computed RMSE based on the specified type.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Calculate squared error
    squared_error = (original - reconstructed) ** 2

    # RMSE per sample
    if metric_type == "sample":
        # Mean squared error over features for each sample, then square root
        rmse = torch.sqrt(squared_error.mean(dim=1))

    # RMSE per feature
    elif metric_type == "feature":
        # Mean squared error over samples for each feature, then square root
        rmse = torch.sqrt(squared_error.mean(dim=0))

    # RMSE total
    elif metric_type == "total":
        # Mean squared error over all samples and features, then square root
        rmse = torch.sqrt(squared_error.mean())

    else:
        raise ValueError(
            f"Invalid rmse_type: {metric_type}. Choose from 'sample', 'feature', 'total'."
        )

    return rmse
