"""Pearson's r (correlation) metric for comparing original and reconstructed data."""

import torch
from torch import Tensor


def compute_pearson_r(original: Tensor, reconstructed: Tensor) -> Tensor:
    """Compute Pearson’s r (correlation) per feature.

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).

    Returns:
        Tensor: Pearson's r for each feature, representing the correlation between
                original and reconstructed values per feature.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Mean of each feature for original and reconstructed
    original_mean = original.mean(dim=0)
    reconstructed_mean = reconstructed.mean(dim=0)

    # Centered values (X - mean)
    original_centered = original - original_mean
    reconstructed_centered = reconstructed - reconstructed_mean

    # Covariance per feature
    covariance = (original_centered * reconstructed_centered).mean(dim=0)

    # Standard deviation per feature
    original_std = original_centered.pow(2).mean(dim=0).sqrt()
    reconstructed_std = reconstructed_centered.pow(2).mean(dim=0).sqrt()

    # Pearson's r per feature
    pearson_r = covariance / (original_std * reconstructed_std)

    # Handle cases where the standard deviation is zero to avoid NaNs
    pearson_r = torch.where(
        (original_std == 0) | (reconstructed_std == 0),
        torch.zeros_like(pearson_r),
        pearson_r,
    )

    return pearson_r
