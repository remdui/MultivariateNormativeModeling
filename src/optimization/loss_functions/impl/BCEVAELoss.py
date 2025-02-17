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
        covariate_embedding_technique: str = "no_embedding",
        cov_dim: int = 0,
    ) -> None:
        """
        Args:

            weight, size_average, reduce, reduction:
                Inherited from PyTorch's _WeightedLoss.
            beta_start (float):
                Initial KL weight at epoch=kl_anneal_start.
            beta_end (float):
                Final KL weight by epoch >= kl_anneal_end.
            kl_anneal_start (int):
                Epoch to start beta annealing.
            kl_anneal_end (int):
                Epoch to complete beta annealing.
            covariate_embedding_technique (str):
                "no_embedding", "input_embedding", "input_feature", "encoder_embedding".
            cov_dim (int):
                # of covariate features if you reconstruct them.
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
        :param recon_x: Reconstructed input.

        :param x: Original data [_, data_dim].
        :param z_mean: [_, latent_dim], for the KL part.
        :param z_logvar: [_, latent_dim], for the KL part.
        :param current_epoch: epoch for beta annealing. If None, use self.beta_end directly.
        :param covariates: [_, cov_dim], if used in reconstruction.
        """
        # 1) Beta for KL
        if current_epoch is not None:
            beta = self._compute_beta(current_epoch)
        else:
            beta = self.beta_end

        # 2) BCE reconstruction
        _, data_dim = x.shape

        if self.covariate_embedding_technique == "no_embedding":
            # 2a) Just data => recon_x is shape [B, data_dim]
            BCE_data = F.binary_cross_entropy(recon_x, x, reduction=self.reduction)
            BCE_cov = 0.0

        elif self.covariate_embedding_technique in {"input_embedding", "input_feature"}:
            # 2b) recon_x is [B, data_dim + cov_dim]. We must split
            if covariates is None:
                raise ValueError(
                    f"covariates must be provided for '{self.covariate_embedding_technique}'"
                )
            recon_data = recon_x[:, :data_dim]
            recon_cov = recon_x[:, data_dim : data_dim + self.cov_dim]

            # BCE for data
            BCE_data = F.binary_cross_entropy(recon_data, x, reduction=self.reduction)
            # BCE for cov
            BCE_cov = F.binary_cross_entropy(
                recon_cov, covariates, reduction=self.reduction
            )

        elif self.covariate_embedding_technique == "encoder_embedding":
            # 2c) Typically recon_x is only data. So do BCE with x
            BCE_data = F.binary_cross_entropy(recon_x, x, reduction=self.reduction)
            BCE_cov = 0.0

        else:
            raise ValueError(
                f"Unknown covariate_embedding_technique: {self.covariate_embedding_technique}"
            )

        bce_loss = BCE_data + BCE_cov

        # 3) KL Divergence
        if self.reduction == "mean":
            # sum across latent dims, then mean over batch
            KLD_per_sample = -0.5 * torch.sum(
                1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1
            )
            KLD = KLD_per_sample.mean()
        else:
            KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        # 4) Weighted final loss
        loss = bce_loss + beta * KLD
        return loss
