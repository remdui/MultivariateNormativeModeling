"""R² Score metric for evaluating the quality of the reconstruction of the data."""

import torch
from torch import Tensor


def compute_r2_score(original: Tensor, reconstructed: Tensor) -> Tensor:
    """Compute the R² Score per feature.

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).

    Returns:
        Tensor: R² score for each feature.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Mean of original data per feature
    original_mean = original.mean(dim=0)

    # Total sum of squares (variance of original data from the mean)
    total_variance: Tensor = ((original - original_mean) ** 2).sum(dim=0)

    # Residual sum of squares (variance of the residuals)
    residual_variance: Tensor = ((original - reconstructed) ** 2).sum(dim=0)

    # Calculate R² per feature
    r2_per_feature = 1 - residual_variance / total_variance

    # Handle cases where total variance is zero to avoid division by zero
    r2_per_feature = torch.where(
        total_variance == 0, torch.zeros_like(r2_per_feature), r2_per_feature
    )

    return r2_per_feature
