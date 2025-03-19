"""Mean Absolute Error (MAE) computation for tensors."""

import torch
from torch import Tensor


def compute_mae(
    original: Tensor,
    reconstructed: Tensor,
    metric_type: str = "total",
) -> Tensor:
    """Compute Mean Absolute Error (MAE) for different types.

    MAE assesses the average absolute difference between original and reconstructed values, providing an interpretable measure of error that is less sensitive to large outliers compared to MSE

    Equation: MAE = 1/n * Σ |original - reconstructed|

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        metric_type (str): Type of MAE to compute. Options are 'sample', 'feature', 'total', 'covariate'.

    Returns:
        Tensor: Computed MAE based on the specified type.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Calculate absolute error
    absolute_error = torch.abs(original - reconstructed)

    # MAE per sample
    if metric_type == "sample":
        # Average over features for each sample
        mae = absolute_error.mean(dim=1)

    # MAE per feature
    elif metric_type == "feature":
        # Average over samples for each feature
        mae = absolute_error.mean(dim=0)

    # MAE total
    elif metric_type == "total":
        # Average over all samples and features
        mae = absolute_error.mean()

    else:
        raise ValueError(
            f"Invalid mae_type: {metric_type}. Choose from 'sample', 'feature', 'total', 'covariate'."
        )

    return mae
