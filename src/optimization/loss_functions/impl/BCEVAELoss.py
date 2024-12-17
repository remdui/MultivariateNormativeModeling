"""BCEVAELoss class implementation for VAE with beta-VAE and KL annealing."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss


class BCEVAELoss(_WeightedLoss):
    """Binary Cross Entropy + KL Divergence loss for VAE with beta-VAE and KL annealing."""

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: Any = None,
        reduce: Any = None,
        reduction: str = "mean",
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        kl_anneal_start: int = 0,
        kl_anneal_end: int = 0,
    ) -> None:
        """Initialize the BCEVAELoss class with beta-VAE and KL annealing parameters.

        Args:
            weight (Tensor | None): Weight tensor for weighted loss.
            size_average (Any): Deprecated (use reduction).
            reduce (Any): Deprecated (use reduction).
            reduction (str): Specifies the reduction to apply to the output.
            beta_start (float): Initial value of beta. Default is 0.0.
            beta_end (float): Final value of beta after annealing. Default is 1.0.
            kl_anneal_start (int): The epoch at which to start annealing beta. Default is 0.
            kl_anneal_end (int): The epoch at which beta should reach its final value (beta_end). Default is 0.
        """
        super().__init__(weight, size_average, reduce, reduction)

        # Store beta VAE parameters
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.kl_anneal_start = kl_anneal_start
        self.kl_anneal_end = kl_anneal_end

    def _compute_beta(self, current_epoch: int) -> float:
        """Compute the current beta value given the current epoch, implementing a linear annealing schedule.

        Args:
            current_epoch (int): The current epoch in training.

        Returns:
            float: The beta value for the given epoch.
        """
        # If no annealing is specified or kl_anneal_end <= kl_anneal_start, just return beta_end
        if self.kl_anneal_end <= self.kl_anneal_start:
            return self.beta_end

        # If we are before the start of annealing, return beta_start
        if current_epoch < self.kl_anneal_start:
            return self.beta_start

        # If we are after the end of annealing, return beta_end
        if current_epoch >= self.kl_anneal_end:
            return self.beta_end

        # Otherwise, linearly interpolate beta between beta_start and beta_end
        progress = (current_epoch - self.kl_anneal_start) / (
            self.kl_anneal_end - self.kl_anneal_start
        )
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def forward(
        self,
        recon_x: Tensor,
        x: Tensor,
        z_mean: Tensor,
        z_logvar: Tensor,
        current_epoch: int | None = None,
    ) -> Tensor:
        """Binary Cross Entropy + beta * KL Divergence loss for VAE.

        Args:
            recon_x (Tensor): Reconstructed input.
            x (Tensor): Input data.
            z_mean (Tensor): Mean of the latent space.
            z_logvar (Tensor): Log variance of the latent space.
            current_epoch (int | None): Current epoch number for KL annealing. If None, beta_end is used.

        Returns:
            Tensor: The loss value.
        """
        # Compute the current beta if current_epoch is provided
        if current_epoch is not None:
            beta = self._compute_beta(current_epoch)
        else:
            beta = self.beta_end  # If no epoch is given, default to final beta

        # Compute the Binary Cross Entropy (BCE) term
        BCE = F.binary_cross_entropy(recon_x, x, reduction=self.reduction)

        # Compute the KL divergence term
        # sum over latent dimensions
        if self.reduction == "mean":
            # For mean reduction, we compute the KL across each latent dimension and then average
            KLD_per_sample = -0.5 * torch.sum(
                1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1
            )
            KLD = KLD_per_sample.mean()
        else:
            # 'sum' or 'none' cases. If 'sum', we sum over entire batch's latent dimensions.
            KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        # Apply beta weighting to the KL divergence term
        loss = BCE + beta * KLD

        return loss
