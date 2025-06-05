"""
Disentangle embedding strategy: partition the latent space into.

a sensitive-specific portion and an invariant portion.
"""

import torch
from torch import Tensor, nn

from entities.properties import Properties
from model.models.covariates.base_embedding_strategy import BaseEmbeddingStrategy
from model.models.util.vae import reparameterize
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


class DisentangleEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    DisentangleEmbeddingStrategy for partitioning latent space into:

      - z_s ∈ ℝ^{cov_dim}: “sensitive” subspace, used to predict covariates
      - z_u ∈ ℝ^{latent_dim - cov_dim}: “uninformative” subspace, forced to be independent of covariates

    Reconstruction is done from (z_u, covariates) only.
    Classification/regression of covariates is done from z_s.
    """

    def __init__(
        self,
        vae: any,
        covariate_info: dict,
        hsic_lambda: float = 1.0,
    ) -> None:
        """
        Args:

            vae: The VAE model instance (with encoder & decoder).
            covariate_info: Dict containing:
              - "labels": list of all covariate names (for display—unused directly here).
              - "continuous": list of indices of continuous covariate columns.
              - "categorical": dict mapping group_name → list of one‐hot column indices.
            hsic_lambda: Weight for the HSIC penalty on z_u vs covariates.
        """
        super().__init__(vae)

        device = Properties.get_instance().system.device

        self.covariate_info = covariate_info or {"labels": []}
        self.continuous_indices = self.covariate_info.get("continuous", [])
        self.categorical_groups = self.covariate_info.get("categorical", {})

        self.cov_dim = len(self.continuous_indices) + sum(
            len(v) for v in self.categorical_groups.values()
        )

        self.sensitive_dim = self.cov_dim
        self.hsic_lambda = hsic_lambda

        self.cont_predictor: nn.Linear | None = None
        if self.continuous_indices:
            self.cont_predictor = nn.Linear(
                self.sensitive_dim, len(self.continuous_indices)
            ).to(device)

        self.cat_predictors = nn.ModuleDict()
        for group_name, indices in self.categorical_groups.items():
            num_classes = len(indices)
            self.cat_predictors[group_name] = nn.Linear(
                self.sensitive_dim, num_classes
            ).to(device)

    def configure_dimensions(
        self, input_dim: int, output_dim: int, cov_dim: int, latent_dim: int
    ) -> dict:
        """Define how encoder & decoder dims should be set."""
        return {
            "encoder_input_dim": input_dim,
            "encoder_output_dim": latent_dim + self.sensitive_dim,
            "decoder_input_dim": latent_dim + self.sensitive_dim,
            "decoder_output_dim": output_dim,
        }

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict:
        """Forward pass."""
        if covariates is None:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                "Covariates must be provided for disentangle embedding."
            )

        z_mean, z_logvar = self.vae.encoder(x)
        z = reparameterize(z_mean, z_logvar)

        if z.size(1) < self.sensitive_dim:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                f"Latent dimension {z.size(1)} is smaller than required sensitive_dim={self.sensitive_dim}."
            )
        z_s = z[:, : self.sensitive_dim]
        z_u = z[:, self.sensitive_dim :]

        decoder_input = torch.cat([z_u, covariates], dim=1)
        x_recon = self.vae.decoder(decoder_input)

        disc_preds: dict[str, Tensor] = {}
        if self.cont_predictor is not None:
            cont_pred = self.cont_predictor(z_s)
            disc_preds["continuous"] = cont_pred

        for group_name, predictor in self.cat_predictors.items():
            disc_preds[group_name] = predictor(z_s)

        return {
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
            "disc_preds": disc_preds,
        }
