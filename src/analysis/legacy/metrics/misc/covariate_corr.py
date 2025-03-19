"""Functions to evaluate the correlation between reconstruction error and covariates."""

from collections.abc import Callable

import torch

from analysis.legacy.metrics.stats.pearson_r import compute_pearson_r


def evaluate_covariates_correlation(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    covariates: torch.Tensor,
    metric_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """
    Analyze the correlation between reconstruction error and multiple covariates.

    Args:
        original (torch.Tensor): Original data tensor (num_samples, num_features).
        reconstructed (torch.Tensor): Reconstructed data tensor (num_samples, num_features).
        covariates (torch.Tensor): Covariate values (num_samples, num_covariates).
        metric_fn (callable): Function to compute reconstruction error. Must accept
                              'metric_type' as a keyword argument and support 'sample'.

    Returns:
        torch.Tensor: Tensor of correlation values for each covariate.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")
    if covariates.shape[0] != original.shape[0]:
        raise ValueError("Number of covariates must match the number of samples.")

    # Verify the metric function supports the `metric_type` argument
    if "metric_type" not in metric_fn.__code__.co_varnames:
        raise ValueError("The provided metric function does not support 'metric_type'.")

    # Compute reconstruction error using the provided metric function with 'metric_type=sample'
    reconstruction_error = metric_fn(original, reconstructed, metric_type="sample")

    # Compute correlations for each covariate using Pearson correlation
    correlations = []
    for cov_idx in range(covariates.shape[1]):
        covariate = covariates[:, cov_idx]
        corr = compute_pearson_r(reconstruction_error, covariate)
        correlations.append(corr)

    # Convert correlations to a tensor
    return torch.stack(correlations)
