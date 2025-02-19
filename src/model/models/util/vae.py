"""VAE specific utility functions module."""

import torch
from torch import Tensor


def reparameterize(z_mean: Tensor, z_logvar: Tensor) -> Tensor:
    """
    Reparameterize the latent space using the reparameterization trick.

    This function samples from the latent distribution defined by z_mean and z_logvar,
    allowing for backpropagation through the stochastic sampling process.

    Args:
        z_mean (Tensor): Mean of the latent distribution.
        z_logvar (Tensor): Log variance of the latent distribution.

    Returns:
        Tensor: A sample from the latent space.
    """
    std = torch.exp(0.5 * z_logvar)
    eps = torch.randn_like(std)  # Sample epsilon ~ N(0, 1) with the same shape as std.
    return z_mean + std * eps
