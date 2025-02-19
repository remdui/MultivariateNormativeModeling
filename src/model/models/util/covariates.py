"""Utilities for retrieving model embedding techniques and covariate details."""

from entities.log_manager import LogManager
from entities.properties import Properties


def get_embedding_technique() -> str:
    """
    Retrieves the covariate embedding technique defined in the model configuration.

    Returns:
        str: The embedding technique used (e.g., 'no_embedding', 'encoder_embedding').

    Raises:
        KeyError: If the model architecture key is missing in the components dictionary.
    """
    logger = LogManager.get_logger(__name__)
    properties = Properties.get_instance()

    model_components = properties.model.components.get(
        properties.model.architecture, None
    )

    if model_components is None:
        logger.warning(
            f"Model architecture '{properties.model.architecture}' not found in components."
        )
        return "no_embedding"

    return model_components.get("covariate_embedding", "no_embedding")


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
