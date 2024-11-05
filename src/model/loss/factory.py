"""Factory module for creating loss function instances."""

from collections.abc import Callable

from model.loss.functions import bce_kld_loss, mse_kld_loss

# Mapping for available loss functions
LOSS_FUNCTION_MAPPING: dict[str, Callable] = {
    "bce_kld": bce_kld_loss,
    "mse_kld": mse_kld_loss,
}


def get_loss_function(loss_type: str) -> Callable:
    """Factory method to get the loss function based on config.

    Args:
        loss_type (str): The type of loss function (e.g., 'bce_kld', 'mse_kld').

    Returns:
        Callable: The loss function.

    Raises:
        ValueError: If the loss function type is not supported.
    """
    loss_function = LOSS_FUNCTION_MAPPING.get(loss_type.lower())
    if not loss_function:
        raise ValueError(f"Unknown loss function type: {loss_type}")
    return loss_function
