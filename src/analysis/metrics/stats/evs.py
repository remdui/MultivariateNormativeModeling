"""Explained Variance Score (EVS) metric for evaluating reconstruction quality."""

import torch
from torch import Tensor


def compute_evs(original: Tensor, reconstructed: Tensor) -> Tensor:
    """Compute the Explained Variance Score (EVS) per feature.

    EVS measures how much of the variance in each feature is captured by the reconstruction, indicating the model's effectiveness in preserving variability without focusing on the direction of errors
    Note: Similar to R², but does not penalize for the direction of errors

    Equation: EVS = 1 - Var(residuals) / Var(original)

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).

    Returns:
        Tensor: Explained Variance Score for each feature.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Compute variance of original data per feature
    original_variance = original.var(dim=0, unbiased=False)

    # Compute variance of residuals (reconstruction error) per feature
    residuals_variance = (original - reconstructed).var(dim=0, unbiased=False)

    # Calculate explained variance score per feature
    explained_variance = 1 - residuals_variance / original_variance

    # Handle cases where original variance is zero
    explained_variance = torch.where(
        original_variance == 0, torch.zeros_like(explained_variance), explained_variance
    )

    return explained_variance
