"""Regularization functions for model parameters."""

from torch import Tensor, nn


def l2_regularization(model: nn.Module, lambda_reg: float) -> Tensor:
    """Apply L2 regularization to model parameters."""
    l2_reg: Tensor = Tensor([0])
    for param in model.parameters():
        l2_reg += param.norm(2)
    return lambda_reg * l2_reg
