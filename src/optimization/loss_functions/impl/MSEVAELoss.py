"""Custom loss function for VAE model with beta-VAE and KL annealing."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss


class MSEVAELoss(_WeightedLoss):
    """Mean Squared Error + KL Divergence loss for VAE with beta-VAE and KL annealing options."""

    def __init__(
        self,
        # Standard VAE fields:
        weight: Tensor | None = None,
        size_average: Any = None,
        reduce: Any = None,
        reduction: str = "mean",
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        kl_anneal_start: int = 0,
        kl_anneal_end: int = 0,
        covariate_embedding_technique: str = "no_embedding",
        cov_dim: int = 0,
    ) -> None:
        """Initialize the MSEVAELoss class with beta-VAE, KL annealing, and covariate logic.

        Args:
            weight, size_average, reduce, reduction: inherited from PyTorch _WeightedLoss.
            beta_start (float): initial KL weight.
            beta_end (float): final KL weight.
            kl_anneal_start (int): epoch to begin annealing.
            kl_anneal_end (int): epoch to finish annealing.
            covariate_embedding_technique (str): 'no_embedding', 'input_embedding', 'input_feature', or 'encoder_embedding'.
            cov_dim (int): number of covariate features if we reconstruct them.
        """
        super().__init__(weight, size_average, reduce, reduction)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.kl_anneal_start = kl_anneal_start
        self.kl_anneal_end = kl_anneal_end

        self.covariate_embedding_technique = covariate_embedding_technique
        self.cov_dim = cov_dim

    def _compute_beta(self, current_epoch: int) -> float:
        """Linearly anneal beta from beta_start to beta_end over [kl_anneal_start, kl_anneal_end]."""
        if self.kl_anneal_end <= self.kl_anneal_start:
            return self.beta_end
        if current_epoch < self.kl_anneal_start:
            return self.beta_start
        if current_epoch >= self.kl_anneal_end:
            return self.beta_end

        progress = (current_epoch - self.kl_anneal_start) / float(
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
        covariates: Tensor | None = None,
    ) -> Tensor:
        """
        :param recon_x: Reconstructed data.

        :param x: Original data [_, data_dim].
        :param z_mean: [_, latent_dim], used for KL divergence.
        :param  z_logvar: [_, latent_dim], used for KL divergence.
        :param current_epoch: For KL annealing. If None, use self.beta_end directly.
        :param covariates: [_, cov_dim], or None if not used.
        """
        if current_epoch is not None:
            beta = self._compute_beta(current_epoch)
        else:
            beta = self.beta_end

        _, data_dim = x.shape

        if self.covariate_embedding_technique in {
            "no_embedding",
            "encoder_embedding",
            "decoder_embedding",
        }:
            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)
            mse_cov = 0.0

        elif self.covariate_embedding_technique == "input_feature":
            x = torch.cat([x, covariates], dim=1)
            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)
            mse_cov = 0.0

        elif self.covariate_embedding_technique == "conditional_embedding":
            if covariates is None:
                raise ValueError(
                    f"covariates must be provided for '{self.covariate_embedding_technique}' mode."
                )
            recon_data = recon_x[:, :data_dim]
            recon_cov = recon_x[:, data_dim : data_dim + self.cov_dim]
            mse_data = F.mse_loss(recon_data, x, reduction=self.reduction)
            mse_cov = F.mse_loss(recon_cov, covariates, reduction=self.reduction)

        else:
            raise ValueError(
                f"Unknown covariate_embedding_technique: {self.covariate_embedding_technique}"
            )

        recon_loss = mse_data + mse_cov

        if self.reduction == "mean":
            KLD_per_sample = -0.5 * torch.sum(
                1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1
            )
            KLD = KLD_per_sample.mean()
        else:
            KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        loss = recon_loss + beta * KLD
        return loss
