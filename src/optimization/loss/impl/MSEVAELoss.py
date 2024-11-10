"""Custom loss function for VAE model."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss


class MSEVAELoss(_WeightedLoss):
    """Mean Squared Error + KL Divergence loss for VAE."""

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: Any = None,
        reduce: Any = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize the BCEVAELoss class."""
        super().__init__(weight, size_average, reduce, reduction)

    def forward(
        self, recon_x: Tensor, x: Tensor, z_mean: Tensor, z_logvar: Tensor
    ) -> Tensor:
        """Mean Squared Error + KL Divergence loss for VAE.

        Args:
            recon_x (Tensor): Reconstructed input.
            x (Tensor): Input data.
            z_mean (Tensor): Mean of the latent space.
            z_logvar (Tensor): Log variance of the latent space.

        Returns:
            Tensor: Loss value. Summed over all elements in the batch.
        """
        MSE = F.mse_loss(recon_x, x, reduction=self.reduction)
        KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        if self.reduction == "mean":
            KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
            KLD = KLD.mean()

        return MSE + KLD
