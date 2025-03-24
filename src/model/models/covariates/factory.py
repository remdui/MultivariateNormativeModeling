"""Factory method for retrieving the embedding strategy based on the model configuration."""

from typing import Any

from entities.log_manager import LogManager
from entities.properties import Properties
from model.models.covariates.impl.age_prior_embedding import AgePriorEmbeddingStrategy
from model.models.covariates.impl.conditional_vae_embedding import (
    ConditionalEmbeddingStrategy,
)
from model.models.covariates.impl.decoder_embedding import DecoderEmbeddingStrategy
from model.models.covariates.impl.encoder_embedding import EncoderEmbeddingStrategy
from model.models.covariates.impl.input_feature_embedding import (
    InputFeatureEmbeddingStrategy,
)
from model.models.covariates.impl.no_embedding import NoEmbeddingStrategy
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


def get_embedding_strategy(vae: Any) -> Any:
    """Factory method for retrieving the embedding strategy based on the model configuration."""
    logger = LogManager.get_logger(__name__)
    technique = get_embedding_technique()
    logger.info(f"Initializing VAE using {technique} embedding.")

    if technique == "age_prior_embedding":
        return AgePriorEmbeddingStrategy(vae)
    if technique == "no_embedding":
        return NoEmbeddingStrategy(vae)
    if technique == "input_feature":
        return InputFeatureEmbeddingStrategy(vae)
    if technique == "conditional_embedding":
        return ConditionalEmbeddingStrategy(vae)
    if technique == "encoder_embedding":
        return EncoderEmbeddingStrategy(vae)
    if technique == "decoder_embedding":
        return DecoderEmbeddingStrategy(vae)
    raise UnsupportedCovariateEmbeddingTechniqueError(
        f"Unknown covariate_embedding technique: {technique}"
    )


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
