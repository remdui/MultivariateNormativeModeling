"""Factory method for retrieving the embedding strategy based on the model configuration."""

from typing import Any

from entities.log_manager import LogManager
from entities.properties import Properties
from model.models.covariates.impl.adversarial_embedding import (
    SimpleAdversarialEmbeddingStrategy,
)
from model.models.covariates.impl.conditional_adversarial_embedding import (
    SimpleConditionalAdversarialEmbeddingStrategy,
)
from model.models.covariates.impl.conditional_embedding import (
    ConditionalEmbeddingStrategy,
)
from model.models.covariates.impl.decoder_embedding import DecoderEmbeddingStrategy
from model.models.covariates.impl.encoder_embedding import EncoderEmbeddingStrategy
from model.models.covariates.impl.encoderdecoder_embedding import (
    EncoderDecoderEmbeddingStrategy,
)
from model.models.covariates.impl.fair_embedding import FairEmbeddingStrategy
from model.models.covariates.impl.hsic_embedding import HSICEmbeddingStrategy
from model.models.covariates.impl.input_feature_embedding import (
    InputFeatureEmbeddingStrategy,
)
from model.models.covariates.impl.no_embedding import NoEmbeddingStrategy
from util.errors import UnsupportedCovariateEmbeddingTechniqueError

covariate_info = {
    "labels": ["age", "sex"],
    "continuous": [0],
    "categorical": {"sex": [1, 2]},
}

# covariate_info = {
#     "labels": ["age", "sex", "site"],
#     "continuous": [0],
#     "categorical": {"sex": [1, 2], "site": [3, 4, 5], },
# }


def get_embedding_strategy(vae: Any) -> Any:
    """Factory method for retrieving the embedding strategy based on the model configuration."""
    logger = LogManager.get_logger(__name__)
    technique = get_embedding_technique()
    logger.info(f"Initializing VAE using {technique} embedding.")

    if technique == "no_embedding":
        return NoEmbeddingStrategy(vae)
    if technique == "input_feature_embedding":
        return InputFeatureEmbeddingStrategy(vae)
    if technique == "conditional_embedding":
        return ConditionalEmbeddingStrategy(vae)
    if technique == "encoderdecoder_embedding":
        return EncoderDecoderEmbeddingStrategy(vae)
    if technique == "encoder_embedding":
        return EncoderEmbeddingStrategy(vae)
    if technique == "decoder_embedding":
        return DecoderEmbeddingStrategy(vae)
    if technique == "adversarial_embedding":
        return SimpleAdversarialEmbeddingStrategy(
            vae, lambda_adv=1.0, covariate_info=covariate_info
        )

    if technique == "conditional_adversarial_embedding":
        return SimpleConditionalAdversarialEmbeddingStrategy(
            vae, lambda_adv=1.0, covariate_info=covariate_info
        )

    if technique == "fair_embedding":
        return FairEmbeddingStrategy(vae, mmd_lambda=1.0, covariate_info=covariate_info)

    if technique == "hsic_embedding":
        return HSICEmbeddingStrategy(vae, hsic_lambda=1.0)

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
