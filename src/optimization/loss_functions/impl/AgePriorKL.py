"""Custom loss function."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss

from optimization.loss_functions.util.kl_annealing import KLAnnealing


class AgePriorKL(_WeightedLoss):
    """Custom loss function."""

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
        """
        Args:

            weight, size_average, reduce, reduction: inherited from PyTorch _WeightedLoss.
            beta_start (float): initial KL weight.
            beta_end (float): final KL weight.
            kl_anneal_start (int): epoch to begin annealing.
            kl_anneal_end (int): epoch to finish annealing.
            covariate_embedding_technique (str): the technique used for embedding covariates.
            cov_dim (int): number of covariate features if we reconstruct them.
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
        prior_mu = model_outputs.get("prior_mu", None)
        prior_logvar = model_outputs.get("prior_logvar", None)

        if current_epoch is not None:
            beta = self.kl_annealer.compute_beta(current_epoch)
        else:
            beta = self.kl_annealer.beta_end

        # x = torch.cat([x, covariates], dim=1)
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

        total_loss = recon_loss + beta * kl_div

        return total_loss
