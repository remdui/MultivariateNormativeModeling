"""BCEVAELoss class implementation for VAE with beta-VAE and KL annealing."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss

from optimization.loss_functions.util.kl_annealing import KLAnnealing


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
               <>
            cov_dim (int):
                # of covariate features if you reconstruct them.
        """
        super().__init__(weight, size_average, reduce, reduction)

        self.kl_annealer = KLAnnealing(
            beta_start=beta_start,
            beta_end=beta_end,
            kl_anneal_start=kl_anneal_start,
            kl_anneal_end=kl_anneal_end,
        )
        self.covariate_embedding_technique = covariate_embedding_technique
        self.cov_dim = cov_dim

    def forward(
        self,
        model_outputs: dict[str, Tensor],
        x: Tensor,
        covariates: Tensor | None = None,
        current_epoch: int | None = None,
    ) -> Tensor:
        """
        Compute the VAE-style loss (e.g., reconstruction + KL) given the model outputs and input data.

        This method expects a dictionary of model outputs, which may contain entries such as:
        - "x_recon": The reconstructed input (torch.Tensor).
        - "z_mean" and "z_logvar": The posterior mean/log-variance for the latent variable.
        - "z": A sampled latent vector.
        - "prior_mu" and "prior_logvar": (Optional) Mean and log-variance of a learned conditional prior,
                                         if the model implements age- or covariate-based priors.

        Args:
            model_outputs (dict[str, torch.Tensor]): Dictionary from the model's forward pass,
                containing any relevant intermediate or final tensors needed for the loss.
            x (torch.Tensor): The original input data (e.g., features or images) for calculating
                reconstruction error.
            covariates (torch.Tensor | None, optional): Additional covariate data (such as age or sex),
                which may be used by some model variants. Defaults to None if not required.
            current_epoch (int | None, optional): Current training epoch for scheduling or annealing
                purposes (e.g., gradually increasing KL weight). Defaults to None if not used.

        Returns:
            torch.Tensor: A scalar loss value that typically combines a reconstruction term
            (e.g., MSE or BCE) and one or more KL terms (standard or conditional prior).
        """
        recon_x = model_outputs["x_recon"]
        z_mean = model_outputs.get("z_mean", None)
        z_logvar = model_outputs.get("z_logvar", None)

        if current_epoch is not None:
            beta = self.kl_annealer.compute_beta(current_epoch)
        else:
            beta = self.kl_annealer.beta_end

        _, data_dim = x.shape

        if self.covariate_embedding_technique == "no_embedding":
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction=self.reduction)
            cov_loss = 0.0

        elif self.covariate_embedding_technique == "input_feature":
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
            raise ValueError(
                f"Unknown covariate_embedding_technique: {self.covariate_embedding_technique}"
            )

        bce_loss = recon_loss + cov_loss

        if self.reduction == "mean":
            kld_per_sample = -0.5 * torch.sum(
                1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1
            )
            kld = kld_per_sample.mean()
        else:
            kld = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        loss = bce_loss + beta * kld

        return loss
