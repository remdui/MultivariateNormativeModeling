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

from optimization.loss_functions.impl.AgePriorKL import AgePriorKL
from optimization.loss_functions.impl.BCEVAELoss import BCEVAELoss
from optimization.loss_functions.impl.MSEVAELoss import MSEVAELoss

# Type alias for loss function classes (subclasses of torch.nn.Module)
LossFunctionClass = type[Module]

# Mapping for available loss functions (private)
_LOSS_FUNCTION_MAPPING: dict[str, LossFunctionClass] = {
    # Custom loss function implementations
    "bce_vae": BCEVAELoss,
    "mse_vae": MSEVAELoss,
    "age_prior_kl": AgePriorKL,
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
    """
    Factory method to create a loss function instance based on configuration.

    Args:
        loss_type (str): The type of loss function (e.g., 'bce_vae', 'mse', 'ctc').
                         The lookup is case-insensitive.
        *args: Additional positional arguments for the loss function's constructor.
        **kwargs: Additional keyword arguments for the loss function's constructor.

    Returns:
        Module: An instance of the specified loss function.

    Raises:
        ValueError: If the loss function type is not supported.
    """
    loss_function_class = _LOSS_FUNCTION_MAPPING.get(loss_type.lower())
    if loss_function_class is None:
        raise ValueError(f"Unknown loss function type: {loss_type}")
    return loss_function_class(*args, **kwargs)
