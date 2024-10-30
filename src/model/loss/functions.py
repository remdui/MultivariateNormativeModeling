"""Loss functions for VAE model."""

import torch.nn.functional as F


def bce_kld_loss(recon_x, x, mu, logvar):
    """Binary Cross Entropy + KL Divergence loss."""
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    return BCE + KLD


def mse_kld_loss(recon_x, x, mu, logvar):
    """Mean Squared Error + KL Divergence loss."""
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    return MSE + KLD
