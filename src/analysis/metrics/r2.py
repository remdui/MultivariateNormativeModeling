"""R² Score for data reconstruction."""

import torch
from torch import Tensor


def compute_r2_score(
    original: Tensor, reconstructed: Tensor, metric_type: str = "feature"
) -> Tensor:
    """
    Compute the R² Score for data reconstruction.

    R² quantifies the proportion of variance in the original data that is captured by the reconstruction,
    providing a measure of explained variability.

    Equation (per feature): R² = 1 - Var(residuals) / Var(original)

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        metric_type (str): Specify "feature" for R² per feature or "total" for combined R² score.

    Returns:
        Tensor: R² score(s). A tensor of shape (num_features,) for "feature" or a scalar for "total".
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Mean of original data per feature
    original_mean = original.mean(dim=0)

    # Total sum of squares (variance of original data from the mean)
    total_variance: Tensor = ((original - original_mean) ** 2).sum(dim=0)

    # Residual sum of squares (variance of the residuals)
    residual_variance: Tensor = ((original - reconstructed) ** 2).sum(dim=0)

    if metric_type == "feature":
        # Calculate R² per feature
        r2_per_feature = 1 - residual_variance / total_variance

        # Handle cases where total variance is zero to avoid division by zero
        r2_per_feature = torch.where(
            total_variance == 0, torch.zeros_like(r2_per_feature), r2_per_feature
        )

        return r2_per_feature

    if metric_type == "total":
        # Aggregate variances across all features for a total R² score
        total_variance_sum = total_variance.sum()
        residual_variance_sum = residual_variance.sum()

        # Calculate total R²
        r2_total = 1 - residual_variance_sum / total_variance_sum

        # Handle cases where total variance sum is zero to avoid division by zero
        r2_total = torch.tensor(0.0) if total_variance_sum == 0 else r2_total

        return r2_total

    raise ValueError('Invalid metric_type. Choose either "feature" or "total".')
