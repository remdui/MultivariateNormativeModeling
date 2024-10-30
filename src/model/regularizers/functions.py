"""Regularization functions for model parameters."""


def l2_regularization(model, lambda_reg):
    """Apply L2 regularization to model parameters."""
    l2_reg = 0
    for param in model.parameters():
        l2_reg += param.norm(2)
    return lambda_reg * l2_reg
