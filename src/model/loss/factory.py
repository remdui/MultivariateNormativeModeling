"""Factory method to get the loss function based on config."""

from collections.abc import Callable

from model.loss.functions import bce_kld_loss, mse_kld_loss


def get_loss_function(loss_type: str) -> Callable:
    """Factory method to get the loss function based on config."""
    if loss_type == "bce_kld":
        return bce_kld_loss
    if loss_type == "mse_kld":
        return mse_kld_loss
    raise ValueError(f"Unknown loss function type: {loss_type}")
