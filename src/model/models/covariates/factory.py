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
from model.models.covariates.impl.disentangle_embedding import (
    DisentangleEmbeddingStrategy,
)
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


def get_embedding_strategy(vae: Any, covariate_labels: list[str] | None = None) -> Any:
    """Factory method for retrieving the embedding strategy based on the model configuration."""
    logger = LogManager.get_logger(__name__)
    technique = get_embedding_technique()
    covariate_info = build_covariate_info(covariate_labels or [])
    logger.info(f"Initializing VAE using {technique} embedding.")
    logger.info(f"Resolved covariate info: {covariate_info}")

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

    if technique == "disentangle_embedding":
        return DisentangleEmbeddingStrategy(
            vae, covariate_info=covariate_info, hsic_lambda=1.0
        )

    raise UnsupportedCovariateEmbeddingTechniqueError(
        f"Unknown covariate_embedding technique: {technique}"
    )


def build_covariate_info(covariate_labels: list[str]) -> dict[str, Any]:
    """
    Build covariate metadata from encoded covariate labels and encoding config.

    The returned dict is used by embedding strategies that distinguish continuous and
    one-hot categorical covariates.
    """
    properties = Properties.get_instance()
    one_hot_roots = _get_configured_one_hot_roots()
    labels = [label for label in covariate_labels if label]

    categorical: dict[str, list[int]] = {}
    continuous: list[int] = []

    for idx, label in enumerate(labels):
        matched_group = None

        for root in one_hot_roots:
            if label == root or label.startswith(f"{root}_"):
                matched_group = root
                break

        if matched_group is None:
            matched_group = _infer_group_by_prefix(
                label=label,
                labels=labels,
                configured_covariates=properties.dataset.covariates,
            )

        if matched_group is None:
            continuous.append(idx)
        else:
            categorical.setdefault(matched_group, []).append(idx)

    return {
        "labels": labels,
        "continuous": continuous,
        "categorical": categorical,
    }


def _get_configured_one_hot_roots() -> list[str]:
    """Get configured one-hot covariate roots from EncodingTransform settings."""
    properties = Properties.get_instance()
    transforms = properties.dataset.transforms or []

    for transform in transforms:
        if transform.name == "EncodingTransform" and transform.type == "preprocessing":
            configured = transform.params.get("one_hot_encoding", [])
            if isinstance(configured, list):
                return [str(item) for item in configured if str(item).strip()]
    return []


def _infer_group_by_prefix(
    label: str, labels: list[str], configured_covariates: list[str]
) -> str | None:
    """
    Infer one-hot grouping from shared prefixes when explicit config is absent.

    A label is treated as categorical only when at least two encoded columns share
    the same prefix and that prefix is a configured covariate (if provided).
    """
    if "_" not in label:
        return None

    prefix = label.split("_", maxsplit=1)[0]
    matching = [name for name in labels if name.startswith(f"{prefix}_")]
    if len(matching) < 2:
        return None

    if configured_covariates and prefix not in configured_covariates:
        return None

    return prefix


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
