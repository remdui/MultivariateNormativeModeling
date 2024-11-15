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
    # Check that the input and reconstructed data have the same shape
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape")

    # Compute covariance matrix of the input data across features
    cov = torch.cov(original.T)  # Transpose to get covariance over features

    # Add small regularization term to the covariance matrix for stability
    cov += torch.eye(cov.size(0)) * 1e-6

    # Invert the covariance matrix
    cov_inv = torch.inverse(cov)

    # Compute the Mahalanobis distance for each pair
    distances = []
    for i in range(original.size(0)):
        diff = reconstructed[i] - original[i]
        distance = torch.sqrt(torch.dot(diff, torch.mv(cov_inv, diff)))
        distances.append(distance)

    return torch.tensor(distances)
