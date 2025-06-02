"""
HSIC‐constrained VAE covariate embedding strategy.

Implements the “HSIC‐VAE” idea from
“Information Constraints on Auto‐Encoding Variational Bayes” (ICLR 2016).
We encode [x, s] → z, decode [z, s] → x_recon, and in the loss we add
an HSIC penalty between z and s to encourage independence.
"""

import torch
from torch import Tensor

from model.models.covariates.base_embedding_strategy import BaseEmbeddingStrategy
from model.models.util.vae import reparameterize
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


class HSICEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    HSIC‐VAE covariate embedding strategy.

    - Encoder takes [x, covariates] → (z_mean, z_logvar)
    - Sample z = reparameterize(z_mean, z_logvar)
    - Decoder takes [z, covariates] → x_recon
    - During loss, an HSIC penalty is computed between z and covariates
      to enforce that z is independent of s.
    """

    def __init__(
        self,
        vae: any,
        hsic_lambda: float = 1.0,
    ) -> None:
        """
        Args:

            vae: The VAE model instance (with .encoder and .decoder).
            hsic_lambda: Weight for the HSIC penalty term in the loss.
        """
        super().__init__(vae)
        self.hsic_lambda = hsic_lambda

    def configure_dimensions(
        self,
        input_dim: int,
        output_dim: int,
        cov_dim: int,
        latent_dim: int,
    ) -> dict:
        """
        In HSIC‐VAE, we condition both encoder and decoder on s:

          encoder_input_dim  = input_dim + cov_dim
          encoder_output_dim = latent_dim
          decoder_input_dim  = latent_dim + cov_dim
          decoder_output_dim = output_dim
        """
        return {
            "encoder_input_dim": input_dim + cov_dim,
            "encoder_output_dim": latent_dim,
            "decoder_input_dim": latent_dim + cov_dim,
            "decoder_output_dim": output_dim,
        }

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict:
        """
        Forward pass for HSIC embedding:

        Args:
            x: Tensor of shape (batch_size, input_dim)
            covariates: Tensor of shape (batch_size, cov_dim); must not be None.

        Returns:
            A dict containing:
              - "x_recon": reconstructed x (from decoder)
              - "z_mean": mean of posterior
              - "z_logvar": log‐variance of posterior
              - "z": reparameterized sample
        """
        if covariates is None:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                "Covariates must be provided for HSIC embedding."
            )

        encoder_input = torch.cat([x, covariates], dim=1)
        z_mean, z_logvar = self.vae.encoder(encoder_input)
        z = reparameterize(z_mean, z_logvar)
        decoder_input = torch.cat([z, covariates], dim=1)
        x_recon = self.vae.decoder(decoder_input)

        return {
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
        }
