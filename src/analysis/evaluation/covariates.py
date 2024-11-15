"""Functions to evaluate the correlation between reconstruction error and covariates."""

import torch

from analysis.metrics.mae import compute_mae
from analysis.metrics.pearson_r import compute_pearson_r
from analysis.metrics.rmse import compute_rmse


def evaluate_covariates_correlation(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    covariates: torch.Tensor,
    metric: str = "rmse",
    correlation_type: str = "pearson",
) -> torch.Tensor:
    """
    Analyze the correlation between reconstruction error and multiple covariates.

    Args:
        original (torch.Tensor): Original data tensor (num_samples, num_features).
        reconstructed (torch.Tensor): Reconstructed data tensor (num_samples, num_features).
        covariates (torch.Tensor): Covariate values (num_samples, num_covariates).
        metric (str): Metric to compute reconstruction error. Options: 'rmse', 'mae'.
        correlation_type (str): Type of correlation to compute. Options: 'pearson', 'spearman'.

    Returns:
        torch.Tensor: Tensor of correlation values for each covariate.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")
    if covariates.shape[0] != original.shape[0]:
        raise ValueError("Number of covariates must match the number of samples.")

    # Compute reconstruction error
    if metric == "rmse":
        reconstruction_error = compute_rmse(
            original, reconstructed, metric_type="sample"
        )
    elif metric == "mae":
        reconstruction_error = compute_mae(
            original, reconstructed, metric_type="sample"
        )
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose 'rmse' or 'mae'.")

    # Compute correlations for each covariate
    correlations = []
    for cov_idx in range(covariates.shape[1]):
        covariate = covariates[:, cov_idx]
        if correlation_type == "pearson":
            corr = compute_pearson_r(reconstruction_error, covariate)
        else:
            raise ValueError(
                f"Unsupported correlation_type: {correlation_type}. Choose 'pearson' or 'spearman'."
            )

        correlations.append(corr)

    # Convert correlations to a tensor
    return torch.stack(correlations)
