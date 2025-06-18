"""BAG prediction using encoder-decoder style embedding."""

import torch
from torch import Tensor

from model.models.covariates.base_embedding_strategy import BaseEmbeddingStrategy
from model.models.util.vae import reparameterize
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


class BAGEncoderDecoderEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    Keeps the original latent block (size = total_latent_dim) AND appends.

    a single Brain-Age-Gap dimension g.  Decoder is conditioned on
    brain-age = chronological_age + g.
    """

    def __init__(
        self,
        vae,
        gap_dim: int = 1,
        age_index: int = 0,
    ):
        super().__init__(vae)
        self.latent_size = None
        self.gap_dim = gap_dim
        self.age_index = age_index

    def configure_dimensions(
        self, input_dim: int, output_dim: int, cov_dim: int, latent_dim: int
    ) -> dict:
        """Configure dimensions."""
        self.latent_size = latent_dim
        return {
            "encoder_input_dim": input_dim + cov_dim,
            "encoder_output_dim": latent_dim + self.gap_dim,
            "decoder_input_dim": latent_dim + cov_dim,
            "decoder_output_dim": output_dim,
        }

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict:
        """Forward pass."""
        if covariates is None:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                "Covariates (incl. chronological age) must be provided."
            )

        enc_in = torch.cat([x, covariates], dim=1)
        h_mean, h_logv = self.vae.encoder(enc_in)

        g_mean, z_mean = torch.split(h_mean, [self.gap_dim, self.latent_size], dim=1)
        g_logv, z_logv = torch.split(h_logv, [self.gap_dim, self.latent_size], dim=1)
        # z_mean, g_mean = torch.split(h_mean, [self.latent_size, self.gap_dim], dim=1)
        # z_logv, g_logv = torch.split(h_logv, [self.latent_size, self.gap_dim], dim=1)

        z = reparameterize(z_mean, z_logv)
        g = reparameterize(g_mean, g_logv)

        # chronological age
        age = covariates[:, self.age_index : self.age_index + 1]
        brain_age = age + g

        # replace the age column with brain-age for decoder
        if self.age_index == 0:
            cov_dec = torch.cat([brain_age, covariates[:, 1:]], dim=1)
        else:
            cov_dec = torch.cat(
                [
                    covariates[:, : self.age_index],
                    brain_age,
                    covariates[:, self.age_index + 1 :],
                ],
                dim=1,
            )

        dec_in = torch.cat([z, cov_dec], dim=1)

        x_recon = self.vae.decoder(dec_in)

        return {
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_logvar": z_logv,
            "g_mean": g_mean,
            "g_logvar": g_logv,
            "z": z,
            "g": g,
            "brain_age": brain_age,
        }
