"""Utilities for retrieving model embedding techniques and covariate details."""

from entities.properties import Properties


def get_enabled_covariate_count() -> int:
    """
    Computes the number of covariates that are actively used (excluding skipped ones).

    Returns:
        int: The count of enabled covariates.
    """
    properties = Properties.get_instance()

    return len(properties.dataset.covariates) - len(
        properties.dataset.skipped_covariates
    )
