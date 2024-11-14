"""Stratified metric computation for data stratified by multiple covariates."""

from collections.abc import Callable
from typing import Any

import torch


def compute_stratified_metric(
    metric_fn: Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor],
    original_data: torch.Tensor,
    reconstructed_data: torch.Tensor,
    covariate_data: torch.Tensor,
    covariate_filters: list[dict[str, Any]],
    metric_type: str = "total",
) -> torch.Tensor:
    """
    Computes a specified metric on data stratified by multiple covariate values.

    Args:
        metric_fn (Callable): Metric function that takes original and reconstructed data tensors and metric_type.
        original_data (torch.Tensor): Original data tensor (num_samples, num_features).
        reconstructed_data (torch.Tensor): Reconstructed data tensor (num_samples, num_features).
        covariate_data (torch.Tensor): Covariate data tensor (num_samples, num_covariates).
        covariate_filters (List[Dict[str, Any]]): List of filtering conditions for each covariate.
            Each dictionary should contain:
            - 'index': Index of the covariate to use for filtering.
            - 'values': Values or range of values to filter by (list for categorical, or (min, max) for continuous).
            - 'type': Type of covariate ('categorical' or 'continuous').
        metric_type (str): Type of metric computation ('total', 'sample', 'feature').

    Returns:
        torch.Tensor: Computed metric on the filtered subset of data.

    Raises:
        ValueError: If input conditions are invalid or if filtering results in no data.
    """
    # Initialize mask to True for all samples
    mask = torch.ones(covariate_data.size(0), dtype=torch.bool)

    # Apply each covariate filter to update the mask
    for covariate in covariate_filters:
        covariate_index = covariate.get("index")
        covariate_values = covariate.get("values")
        covariate_type = covariate.get("type")

        if covariate_type == "categorical":
            if not isinstance(covariate_values, list):
                raise ValueError(
                    "For categorical covariates, 'values' should be a list."
                )
            covariate_mask = torch.isin(
                covariate_data[:, covariate_index], torch.tensor(covariate_values)
            )
        elif covariate_type == "continuous" and isinstance(covariate_values, tuple):
            min_val, max_val = covariate_values
            covariate_mask = (covariate_data[:, covariate_index] >= min_val) & (
                covariate_data[:, covariate_index] <= max_val
            )
        else:
            raise ValueError(
                "Invalid covariate_type or covariate_values format in covariate_filters."
            )

        # Combine masks for multiple covariates (logical AND)
        mask &= covariate_mask

    # Apply the mask to filter original, reconstructed, and covariate data
    filtered_original = original_data[mask]
    filtered_reconstructed = reconstructed_data[mask]

    if filtered_original.numel() == 0:
        raise ValueError(
            "Filtering resulted in an empty dataset. Adjust covariate values."
        )

    # Compute and return the metric on the filtered data
    return metric_fn(filtered_original, filtered_reconstructed, metric_type)
