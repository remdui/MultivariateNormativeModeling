"""No covariate embedding strategy."""

from torch import Tensor

from model.models.covariates.base_embedding_strategy import BaseEmbeddingStrategy
from model.models.util.vae import reparameterize


class NoEmbeddingStrategy(BaseEmbeddingStrategy):
    """No covariate embedding strategy."""

    def configure_dimensions(
        self, input_dim: int, output_dim: int, cov_dim: int, latent_dim: int
    ) -> dict:
        return {
            "encoder_input_dim": input_dim,
            "encoder_output_dim": latent_dim,
            "decoder_input_dim": latent_dim,
            "decoder_output_dim": output_dim,
        }

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict:
        z_mean, z_logvar = self.vae.encoder(x)
        z = reparameterize(z_mean, z_logvar)
        x_recon = self.vae.decoder(z)
        return {
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
        }
