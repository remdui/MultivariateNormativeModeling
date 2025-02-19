"""Variational Autoencoder with modular components.

This module implements a Variational Autoencoder (VAE) that supports multiple covariate
embedding techniques and conditional priors. The VAE is built using modular encoder and decoder
components, and employs the reparameterization trick for sampling from the latent space.
"""

import torch
from torch import Tensor

from entities.log_manager import LogManager
from model.components.factory import get_decoder, get_encoder
from model.models.abstract_model import AbstractModel
from model.models.util.covariates import (
    get_embedding_technique,
    get_enabled_covariate_count,
)
from model.models.util.priors import CovariatePriorNet
from model.models.util.vae import reparameterize
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


class VAE(AbstractModel):
    """
    Variational Autoencoder with modular components.

    This VAE supports various covariate embedding techniques and conditional priors.
    The encoder and decoder dimensions are adjusted based on the selected covariate embedding
    technique, and an age-conditional prior network is used if applicable.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the VAE model.

        Sets up the encoder and decoder dimensions based on the covariate embedding technique,
        creates the encoder/decoder using factory methods, and instantiates an age prior network.

        Args:
            input_dim (int): Dimension of the input data.
            output_dim (int): Dimension of the output data.
        """
        super().__init__(input_dim, output_dim)
        self.logger = LogManager.get_logger(__name__)
        self.covariate_embedding_technique = get_embedding_technique()
        self.cov_dim = get_enabled_covariate_count()
        self.logger.info(f"Initializing VAE using {self.covariate_embedding_technique}")

        latent_dim = self.model_components.get("latent_dim")

        # Adjust encoder dimensions based on covariate embedding mode.
        if self.covariate_embedding_technique in {
            "input_feature",
            "conditional_embedding",
            "encoder_embedding",
            # "age_prior_embedding"  # This mode uses additional prior logic.
        }:
            self.encoder_input_dim = self.input_dim + self.cov_dim
            self.encoder_output_dim = latent_dim
        else:
            self.encoder_input_dim = self.input_dim
            self.encoder_output_dim = latent_dim

        # Adjust decoder dimensions based on covariate embedding mode.
        if self.covariate_embedding_technique in {
            "input_feature",
            "conditional_embedding",
            # "age_prior_embedding"  # For this mode, decoder output includes covariate info.
        }:
            self.decoder_input_dim = latent_dim
            self.decoder_output_dim = self.output_dim + self.cov_dim
        elif self.covariate_embedding_technique == "decoder_embedding":
            self.decoder_input_dim = latent_dim + self.cov_dim
            self.decoder_output_dim = self.output_dim
        else:
            self.decoder_input_dim = latent_dim
            self.decoder_output_dim = self.output_dim

        # Initialize age-conditional prior network (used for 'age_prior_embedding' mode).
        self.age_prior_net = CovariatePriorNet(latent_dim, [32, 16], num_covariates=1)

        # Create encoder and decoder using factory methods.
        self.encoder = get_encoder(
            encoder_type=self.model_components.get("encoder"),
            input_dim=self.encoder_input_dim,
            hidden_dims=self.properties.model.hidden_layers,
            latent_dim=self.encoder_output_dim,
        )
        self.decoder = get_decoder(
            decoder_type=self.model_components.get("decoder"),
            latent_dim=self.decoder_input_dim,
            hidden_dims=self.properties.model.hidden_layers[::-1],
            output_dim=self.decoder_output_dim,
        )

    def forward(self, x: Tensor, covariates: Tensor | None = None) -> dict[str, Tensor]:
        """
        Execute the forward pass of the VAE.

        Depending on the selected covariate embedding technique, the forward pass may concatenate
        covariates to the input or process them separately. The output is a dictionary containing
        the reconstructed input, latent variables, and optionally, the parameters of a conditional prior.

        Args:
            x (Tensor): Input data of shape [B, data_dim].
            covariates (Tensor | None): Optional covariate data of shape [B, cov_dim].

        Returns:
            dict[str, Tensor]: Dictionary with keys:
                - "x_recon": Reconstructed input.
                - "z_mean": Posterior mean.
                - "z_logvar": Posterior log-variance.
                - "z": Sampled latent vector.
                - "prior_mu", "prior_logvar": (If using age_prior_embedding) Conditional prior parameters.
        """
        outputs = {}

        if self.covariate_embedding_technique == "age_prior_embedding":
            # For age_prior_embedding, the encoder processes x alone and the age prior is computed from covariates.
            z_mean, z_logvar = self.encoder(x)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)
            prior_mu, prior_logvar = self.age_prior_net(covariates)

            outputs.update(
                {
                    "x_recon": x_recon,
                    "z_mean": z_mean,
                    "z_logvar": z_logvar,
                    "z": z,
                    "prior_mu": prior_mu,
                    "prior_logvar": prior_logvar,
                }
            )

        elif self.covariate_embedding_technique == "no_embedding":
            # Standard VAE: encode and decode without using covariates.
            z_mean, z_logvar = self.encoder(x)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)

            outputs.update(
                {
                    "x_recon": x_recon,
                    "z_mean": z_mean,
                    "z_logvar": z_logvar,
                    "z": z,
                }
            )

        elif self.covariate_embedding_technique in {
            "input_feature",
            "conditional_embedding",
            "encoder_embedding",
        }:
            # For these modes, concatenate covariates with input before encoding.
            if covariates is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    f"Covariates must be provided for {self.covariate_embedding_technique} mode."
                )
            encoder_input = torch.cat([x, covariates], dim=1)
            z_mean, z_logvar = self.encoder(encoder_input)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)

            outputs.update(
                {
                    "x_recon": x_recon,
                    "z_mean": z_mean,
                    "z_logvar": z_logvar,
                    "z": z,
                }
            )

        elif self.covariate_embedding_technique == "decoder_embedding":
            # In decoder_embedding mode, the encoder does not receive covariates,
            # but they are concatenated with z before decoding.
            if covariates is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    f"Covariates must be provided for {self.covariate_embedding_technique} mode."
                )
            z_mean, z_logvar = self.encoder(x)
            z = reparameterize(z_mean, z_logvar)
            decoder_input = torch.cat([z, covariates], dim=1)
            x_recon = self.decoder(decoder_input)

            outputs.update(
                {
                    "x_recon": x_recon,
                    "z_mean": z_mean,
                    "z_logvar": z_logvar,
                    "z": z,
                }
            )

        else:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                f"Unknown covariate_embedding technique: {self.covariate_embedding_technique}"
            )

        return outputs
