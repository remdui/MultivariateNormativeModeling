"""Mahalanobis Distance Metric."""

import torch
from torch import Tensor


def compute_mahalanobis_distance(original: Tensor, reconstructed: Tensor) -> Tensor:
    """
    Compute Mahalanobis Distance for each sample in reconstructed data relative to the original data distribution.

    Mahalanobis Distance is a measure of the distance between a point and a distribution, providing a measure of how well the reconstructed data captures the original data distribution.

    Equation: Mahalanobis Distance = sqrt((x - μ)ᵀΣ⁻¹(x - μ))

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).

    Returns:
        Tensor: Mahalanobis distance for each sample in the reconstructed data.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Calculate residuals
    residuals = original - reconstructed  # Shape: (n_samples, n_features)

    # Compute mean vector of residuals
    mean_residuals = residuals.mean(dim=0)  # Shape: (n_features,)

    # Center the residuals
    centered_residuals = residuals - mean_residuals  # Shape: (n_samples, n_features)

    # Compute covariance matrix of residuals
    cov_matrix = torch.cov(centered_residuals.T)  # Shape: (n_features, n_features)

    # Regularize covariance matrix to avoid singularity
    eps = 1e-6
    cov_matrix += torch.eye(cov_matrix.size(0)) * eps

    # Compute the inverse of the covariance matrix
    cov_inv = torch.inverse(cov_matrix)

    # Compute Mahalanobis distances
    left_term = centered_residuals @ cov_inv  # Shape: (n_samples, n_features)
    mahalanobis_distances = torch.sqrt(
        torch.sum(left_term * centered_residuals, dim=1)
    )  # Shape: (n_samples,)

    return mahalanobis_distances
