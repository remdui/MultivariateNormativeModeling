"""Factory module for creating loss function instances."""

from typing import Any

from torch.nn import (
    BCELoss,
    BCEWithLogitsLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    CTCLoss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    HuberLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    Module,
    MSELoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    PoissonNLLLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
)

from model.loss.impl.BCEVAELoss import BCEVAELoss
from model.loss.impl.MSEVAELoss import MSEVAELoss

# Mapping for available loss functions
LOSS_FUNCTION_MAPPING: dict[str, Any] = {
    # Custom loss function implementations
    "bce_vae": BCEVAELoss,
    "mse_vae": MSEVAELoss,
    # PyTorch loss functions
    "l1": L1Loss,
    "nll": NLLLoss,
    "poisson_nll": PoissonNLLLoss,
    "gaussian_nll": GaussianNLLLoss,
    "kldiv": KLDivLoss,
    "mse": MSELoss,
    "bce": BCELoss,
    "bce_with_logits": BCEWithLogitsLoss,
    "hinge_embedding": HingeEmbeddingLoss,
    "multi_label_margin": MultiLabelMarginLoss,
    "huber": HuberLoss,
    "smooth_l1": SmoothL1Loss,
    "soft_margin": SoftMarginLoss,
    "cross_entropy": CrossEntropyLoss,
    "multi_label_soft_margin": MultiLabelSoftMarginLoss,
    "cosine_embedding": CosineEmbeddingLoss,
    "margin_ranking": MarginRankingLoss,
    "multi_margin": MultiMarginLoss,
    "triplet_margin": TripletMarginLoss,
    "triplet_margin_with_distance": TripletMarginWithDistanceLoss,
    "ctc": CTCLoss,
}


def get_loss_function(loss_type: str, *args: Any, **kwargs: Any) -> Module:
    """Factory method to get the loss function based on config.

    Args:
        loss_type (str): The type of loss function (e.g., 'bce_kld', 'mse_kld').
        *args: Positional arguments for the loss function.
        **kwargs: Additional keyword arguments for loss function initialization.

    Returns:
        Module: The loss function instance.

    Raises:
        ValueError: If the loss function type is not supported.
    """
    loss_function_class = LOSS_FUNCTION_MAPPING.get(loss_type.lower())
    if not loss_function_class:
        raise ValueError(f"Unknown loss function type: {loss_type}")
    return loss_function_class(*args, **kwargs)
