"""KL divergence metric for comparing distributions."""

import torch
from torch import Tensor


def compute_kl_divergence(original: Tensor, reconstructed: Tensor) -> Tensor:
    """
    Compute the KL divergence per feature, assuming Gaussian distributions for original and reconstructed data.

    KL divergence measures the difference between two probability distributions, providing a measure of how well the reconstructed data captures the original data distribution.

    Equation: KL divergence = 0.5 * (log(σ_reconstructed / σ_original) + (σ_original + (μ_original - μ_reconstructed)²) / σ_reconstructed - 1)

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).

    Returns:
        Tensor: KL divergence for each feature.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Calculate mean and variance per feature
    original_mean = original.mean(dim=0)
    original_var = original.var(dim=0, unbiased=False)
    reconstructed_mean = reconstructed.mean(dim=0)
    reconstructed_var = reconstructed.var(dim=0, unbiased=False)

    # Compute KL divergence per feature using the formula for two Gaussian distributions
    kl_div = 0.5 * (
        torch.log(reconstructed_var / original_var)
        + (original_var + (original_mean - reconstructed_mean) ** 2) / reconstructed_var
        - 1
    )

    return kl_div
