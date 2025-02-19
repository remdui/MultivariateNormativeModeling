"""Custom loss function for VAE with beta-VAE and KL annealing (MSE variant).

This module defines the MSEVAELoss class, which computes the loss for a variational autoencoder (VAE)
by combining a Mean Squared Error (MSE) reconstruction term with a KL divergence term. The KL weight is
annealed over training epochs using a KLAnnealing schedule. Covariate embedding options are supported.
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


class MSEVAELoss(_WeightedLoss):
    """
    MSE + KL Divergence loss for VAE with beta-VAE and KL annealing.

    Computes reconstruction loss using MSE and adds a KL divergence penalty weighted by an annealed beta.
    Depending on the covariate embedding technique, the reconstruction loss may be computed directly or
    after processing covariate data.
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
        Initialize the MSEVAELoss.

        Args:
            weight, size_average, reduce, reduction: Inherited from _WeightedLoss.
            beta_start (float): Initial KL weight.
            beta_end (float): Final KL weight.
            kl_anneal_start (int): Epoch to begin annealing.
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
        Compute the VAE loss as the sum of MSE reconstruction loss and a weighted KL divergence term.

        Args:
            model_outputs (dict[str, Tensor]): Dictionary from the model's forward pass containing:
                - "x_recon": The reconstructed input.
                - "z_mean": The posterior mean.
                - "z_logvar": The posterior log-variance.
            x (Tensor): Original input tensor.
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

        if self.covariate_embedding_technique in {
            "no_embedding",
            "encoder_embedding",
            "decoder_embedding",
        }:
            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)
            mse_cov = 0.0
        elif self.covariate_embedding_technique == "input_feature":
            if covariates is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Covariates must be provided for 'input_feature' mode."
                )
            x_combined = torch.cat([x, covariates], dim=1)
            mse_data = F.mse_loss(recon_x, x_combined, reduction=self.reduction)
            mse_cov = 0.0
        elif self.covariate_embedding_technique == "conditional_embedding":
            if covariates is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Covariates must be provided for 'conditional_embedding' mode."
                )
            recon_data = recon_x[:, :data_dim]
            recon_cov = recon_x[:, data_dim : data_dim + self.cov_dim]
            mse_data = F.mse_loss(recon_data, x, reduction=self.reduction)
            mse_cov = F.mse_loss(recon_cov, covariates, reduction=self.reduction)
        else:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                f"Unknown covariate embedding technique: {self.covariate_embedding_technique}"
            )

        recon_loss = mse_data + mse_cov

        if self.reduction == "mean":
            kld_per_sample = -0.5 * torch.sum(
                1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1
            )
            kld = kld_per_sample.mean()
        else:
            kld = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        return recon_loss + beta * kld
