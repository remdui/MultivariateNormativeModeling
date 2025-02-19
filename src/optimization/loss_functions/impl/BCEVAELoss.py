"""Custom loss function for VAE with beta-VAE and KL annealing (BCE variant).

This module defines the BCEVAELoss class, which computes the loss for a VAE by combining a binary
cross entropy (BCE) reconstruction term with a KL divergence term. The KL weight is annealed over
training epochs using a KLAnnealing schedule. Covariate embedding options are supported.
"""

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss

from model.models.util.covariates import (
    get_embedding_technique,
    get_enabled_covariate_count,
)
from optimization.loss_functions.util.kl_annealing import KLAnnealing
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


class BCEVAELoss(_WeightedLoss):
    """
    BCE + KL Divergence loss for VAE with beta-VAE and KL annealing.

    Computes reconstruction loss using binary cross entropy and adds a KL divergence penalty weighted by
    an annealed beta. Depending on the covariate embedding technique, the reconstruction loss may be
    computed on the input alone or split into data and covariate parts.
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
        Initialize the BCEVAELoss.

        Args:
            weight, size_average, reduce, reduction: Parameters inherited from _WeightedLoss.
            beta_start (float): Initial KL weight.
            beta_end (float): Final KL weight.
            kl_anneal_start (int): Epoch to start annealing.
            kl_anneal_end (int): Epoch to finish annealing.
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
        Compute the VAE loss as the sum of BCE reconstruction loss and weighted KL divergence.

        Depending on the covariate embedding technique, the reconstruction loss is computed either directly
        or by splitting the reconstruction into data and covariate components.

        Args:
            model_outputs (dict[str, Tensor]): Dictionary from the model's forward pass containing:
                - "x_recon": Reconstructed input.
                - "z_mean": Posterior mean.
                - "z_logvar": Posterior log-variance.
            x (Tensor): Original input data.
            covariates (Tensor | None, optional): Covariate data required for certain embedding modes.
            current_epoch (int | None, optional): Current epoch for KL annealing; if None, final beta is used.

        Returns:
            Tensor: A scalar loss value.
        """
        recon_x = model_outputs["x_recon"]
        z_mean = model_outputs.get("z_mean")
        z_logvar = model_outputs.get("z_logvar")

        beta = (
            self.kl_annealer.compute_beta(current_epoch)
            if current_epoch is not None
            else self.kl_annealer.beta_end
        )

        _, data_dim = x.shape

        if self.covariate_embedding_technique == "no_embedding":
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction=self.reduction)
            cov_loss = 0.0
        elif self.covariate_embedding_technique == "input_feature":
            if covariates is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Covariates must be provided for 'input_feature' mode."
                )
            recon_data = recon_x[:, :data_dim]
            recon_cov = recon_x[:, data_dim : data_dim + self.cov_dim]
            recon_loss = F.binary_cross_entropy(recon_data, x, reduction=self.reduction)
            cov_loss = F.binary_cross_entropy(
                recon_cov, covariates, reduction=self.reduction
            )
        elif self.covariate_embedding_technique == "encoder_embedding":
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction=self.reduction)
            cov_loss = 0.0
        else:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                f"Unknown covariate_embedding_technique: {self.covariate_embedding_technique}"
            )

        total_recon_loss = recon_loss + cov_loss

        if self.reduction == "mean":
            kld_per_sample = -0.5 * torch.sum(
                1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1
            )
            kld = kld_per_sample.mean()
        else:
            kld = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        return total_recon_loss + beta * kld
