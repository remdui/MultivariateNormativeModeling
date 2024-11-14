"""Mean Absolute Error (MAE) computation for tensors."""

import torch
from torch import Tensor


def compute_mae(
    original: Tensor,
    reconstructed: Tensor,
    mae_type: str = "total",
) -> Tensor:
    """Compute Mean Absolute Error (MAE) for different types.

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        mae_type (str): Type of MAE to compute. Options are 'sample', 'feature', 'total', 'covariate'.

    Returns:
        Tensor: Computed MAE based on the specified type.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Calculate absolute error
    absolute_error = torch.abs(original - reconstructed)

    # MAE per sample
    if mae_type == "sample":
        # Average over features for each sample
        mae = absolute_error.mean(dim=1)

    # MAE per feature
    elif mae_type == "feature":
        # Average over samples for each feature
        mae = absolute_error.mean(dim=0)

    # MAE total
    elif mae_type == "total":
        # Average over all samples and features
        mae = absolute_error.mean()

    else:
        raise ValueError(
            f"Invalid mae_type: {mae_type}. Choose from 'sample', 'feature', 'total', 'covariate'."
        )

    return mae
