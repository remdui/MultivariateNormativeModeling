"""Custom loss function for VAE with age-based prior KL divergence.

This module defines the AgePriorKL loss function, which combines a Mean Squared Error (MSE)
reconstruction loss with a KL divergence term comparing the latent posterior (z_mean, z_logvar)
to a learned conditional prior (prior_mu, prior_logvar). The KL divergence is weighted by a beta
factor that is annealed over training epochs.
"""

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss

from model.models.covariates.factory import get_embedding_technique
from model.models.util.covariates import get_enabled_covariate_count
from optimization.loss_functions.util.kl_annealing import KLAnnealing


class AgePriorKL(_WeightedLoss):
    """
    Loss function for VAE with a conditional (age-based) prior.

    Combines an MSE reconstruction loss with a KL divergence term that measures the difference
    between the latent posterior and a learned conditional prior. The KL term is weighted by an
    annealed beta factor.
    """

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
        """
        Initialize the AgePriorKL loss function.

        Args:
            weight, size_average, reduce, reduction: Parameters inherited from _WeightedLoss.
            beta_start (float): Initial KL weight.
            beta_end (float): Final KL weight.
            kl_anneal_start (int): Epoch to begin KL annealing.
            kl_anneal_end (int): Epoch to finish KL annealing.
        """
        super().__init__(weight, size_average, reduce, reduction)
        self.kl_annealer = KLAnnealing(
            beta_start=beta_start,
            beta_end=beta_end,
            kl_anneal_start=kl_anneal_start,
            kl_anneal_end=kl_anneal_end,
        )
        self.covariate_embedding_technique = get_embedding_technique()
        self.cov_dim = get_enabled_covariate_count()

    def forward(
        self,
        model_outputs: dict[str, Tensor],
        x: Tensor,
        covariates: Tensor | None = None,
        current_epoch: int | None = None,
    ) -> Tensor:
        """
        Compute the VAE loss combining MSE reconstruction loss with a weighted KL divergence term.

        For a conditional prior (e.g., age-based), the KL divergence is computed between the latent posterior
        (z_mean, z_logvar) and a learned prior (prior_mu, prior_logvar). The overall loss is given by:

            loss = MSE_loss + beta * KL_divergence

        Args:
            model_outputs (dict[str, Tensor]): Dictionary from the model's forward pass containing:
                - "x_recon": Reconstructed input.
                - "z_mean": Latent posterior mean.
                - "z_logvar": Latent posterior log-variance.
                - "prior_mu": Prior mean for the conditional prior.
                - "prior_logvar": Prior log-variance for the conditional prior.
            x (Tensor): Original input tensor.
            covariates (Tensor | None, optional): Covariate data (if required). Defaults to None.
            current_epoch (int | None, optional): Current epoch for KL annealing. If None, final beta is used.

        Returns:
            Tensor: A scalar loss value.
        """
        recon_x = model_outputs["x_recon"]
        z_mean = model_outputs.get("z_mean")
        z_logvar = model_outputs.get("z_logvar")
        prior_mu = model_outputs.get("prior_mu")
        prior_logvar = model_outputs.get("prior_logvar")

        beta = (
            self.kl_annealer.compute_beta(current_epoch)
            if current_epoch is not None
            else self.kl_annealer.beta_end
        )

        recon_loss = F.mse_loss(recon_x, x, reduction=self.reduction)

        if self.reduction == "mean":
            kl_per_sample = 0.5 * torch.sum(
                prior_logvar
                - z_logvar
                + (torch.exp(z_logvar) + (z_mean - prior_mu) ** 2)
                / torch.exp(prior_logvar)
                - 1,
                dim=1,
            )
            kl_div = kl_per_sample.mean()
        elif self.reduction == "sum":
            kl_div = 0.5 * torch.sum(
                prior_logvar
                - z_logvar
                + (torch.exp(z_logvar) + (z_mean - prior_mu) ** 2)
                / torch.exp(prior_logvar)
                - 1
            )
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'")

        return recon_loss + beta * kl_div
