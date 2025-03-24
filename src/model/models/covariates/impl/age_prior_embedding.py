"""Age prior covariate embedding strategy."""

from torch import Tensor

from model.models.covariates.base_embedding_strategy import BaseEmbeddingStrategy
from model.models.util.vae import reparameterize
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


class AgePriorEmbeddingStrategy(BaseEmbeddingStrategy):
    """Age prior covariate embedding strategy."""

    def configure_dimensions(
        self, input_dim: int, output_dim: int, cov_dim: int, latent_dim: int
    ) -> dict:
        # For age prior, encoder and decoder do not receive covariates.
        return {
            "encoder_input_dim": input_dim,
            "encoder_output_dim": latent_dim,
            "decoder_input_dim": latent_dim,
            "decoder_output_dim": output_dim,
        }

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict:
        if covariates is None:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                "Covariates must be provided for age_prior embedding."
            )
        z_mean, z_logvar = self.vae.encoder(x)
        z = reparameterize(z_mean, z_logvar)
        x_recon = self.vae.decoder(z)
        prior_mu, prior_logvar = self.vae.age_prior_net(covariates)
        return {
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
        }
