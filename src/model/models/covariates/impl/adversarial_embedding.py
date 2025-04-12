"""Adversarial VAE covariate embedding strategy."""

from typing import Any

import torch
from torch import Tensor, nn

from entities.properties import Properties  # Import to get the device, if needed
from model.models.covariates.base_embedding_strategy import BaseEmbeddingStrategy
from model.models.util.vae import reparameterize
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


class GradientReversalFunction(torch.autograd.Function):
    # pylint: disable=all
    """Gradient reversal layer implementation."""

    @staticmethod
    def forward(ctx: Any, x: Any, lambda_: Any) -> Any:
        """Forward pass through the gradient reversal layer."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        # Reverse the gradient by multiplying with -lambda_
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer implementation."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the gradient reversal layer."""
        return GradientReversalFunction.apply(x, self.lambda_)


class AdversarialNetwork(nn.Module):
    """Adversary network implementation."""

    def __init__(
        self,
        latent_dim: int,
        cov_dim: int,
        continuous_indices: list[int] = None,
        categorical_groups: dict[str, list[int]] = None,
    ):
        """
        Args:

            latent_dim: Dimension of the latent code.
            cov_dim: Total number of covariate features.
            continuous_indices: List of indices indicating continuous covariates.
            categorical_groups: Dict mapping a group name to a list of indices corresponding
                                to one-hot encoded categorical variables.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.cov_dim = cov_dim
        self.continuous_indices = (
            continuous_indices if continuous_indices is not None else []
        )
        self.categorical_groups = (
            categorical_groups if categorical_groups is not None else {}
        )

        # For continuous covariates, use a regression module.
        if self.continuous_indices:
            self.regressor = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, len(self.continuous_indices)),
            )
        else:
            self.regressor = None

        # For categorical covariates, create a classifier per group.
        self.classifiers = nn.ModuleDict()
        for group, indices in self.categorical_groups.items():
            self.classifiers[group] = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, len(indices))
            )

    def forward(self, z: Tensor) -> dict:
        """
        Forward pass through the adversary network.

        Returns a dictionary with predictions for continuous and categorical covariates.
        """
        adv_output = {}
        if self.regressor is not None:
            adv_output["continuous"] = self.regressor(z)
        for group, classifier in self.classifiers.items():
            adv_output[group] = classifier(z)
        return adv_output


###########################################
# Adversarial Embedding Strategy          #
###########################################


class AdversarialEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    Adversarial VAE covariate embedding strategy.

    This strategy augments the standard VAE pipeline with an adversarial branch.
    An adversary network attempts to predict the covariates from the latent space.
    A gradient reversal layer is used so that the encoder is penalized if covariate
    information is recoverable from the latent code, thus enforcing invariance.
    """

    def __init__(
        self, vae: Any, lambda_adv: float = 1.0, covariate_info: dict = None
    ) -> None:
        """
        Args:

            vae: Reference to the VAE instance (to access shared encoder/decoder).
            lambda_adv: Weight factor for the gradient reversal layer.
            covariate_info: Dictionary containing covariate metadata, expected to have keys:
                - "labels": list of all covariate names.
                - "continuous": list of indices for continuous covariates.
                - "categorical": dict mapping a categorical group name to list of indices.
        """
        super().__init__(vae)
        self.lambda_adv = lambda_adv
        self.grl = GradientReversalLayer(lambda_adv)
        self.covariate_info = (
            covariate_info if covariate_info is not None else {"labels": []}
        )
        self.cov_dim = len(self.covariate_info.get("labels", []))
        self.continuous_indices = self.covariate_info.get("continuous", [])
        self.categorical_groups = self.covariate_info.get("categorical", {})
        self.adversary = None

    def configure_dimensions(
        self, input_dim: int, output_dim: int, cov_dim: int, latent_dim: int
    ) -> dict:
        """
        Configures dimensions for the VAE components and initializes the adversary network.

        In this adversarial strategy, the decoder reconstructs only the primary data.
        """
        self.adversary = AdversarialNetwork(
            latent_dim,
            cov_dim,
            continuous_indices=self.continuous_indices,
            categorical_groups=self.categorical_groups,
        )
        properties = Properties.get_instance()
        device = properties.system.device
        self.adversary.to(device)

        return {
            "encoder_input_dim": input_dim,
            "encoder_output_dim": latent_dim,
            "decoder_input_dim": latent_dim,
            "decoder_output_dim": output_dim,
        }

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict:
        """
        Forward pass using the adversarial embedding strategy.

        Args:
            x: Input data tensor.
            covariates: Tensor of covariate values.

        Returns:
            A dictionary with:
              - "x_recon": Reconstruction from the decoder.
              - "z_mean": Mean of the latent distribution.
              - "z_logvar": Log-variance of the latent distribution.
              - "z": Sampled latent vector.
              - "adv_preds": Predictions from the adversary network.
        """
        if covariates is None:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                "Covariates must be provided for adversarial embedding."
            )
        z_mean, z_logvar = self.vae.encoder(x)
        z = reparameterize(z_mean, z_logvar)
        x_recon = self.vae.decoder(z)
        z_reversed = self.grl(z)
        if next(self.adversary.parameters()).device != z.device:
            self.adversary.to(z.device)
        adv_preds = self.adversary(z_reversed)

        return {
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
            "adv_preds": adv_preds,
        }
