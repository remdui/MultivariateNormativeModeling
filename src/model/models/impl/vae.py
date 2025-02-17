"""Variational Autoencoder with modular components."""

import torch
from torch import Tensor

from entities.log_manager import LogManager
from model.components.factory import get_decoder, get_encoder
from model.models.abstract_model import AbstractModel


def reparameterize(z_mean: Tensor, z_logvar: Tensor) -> Tensor:
    """Reparameterize the latent space using the Reparameterization Trick.

       See Section 2.4 of the VAE paper (Kingma & Welling, 2013).

    Args:
        z_mean (Tensor): Mean of the latent space.
        z_logvar (Tensor): Log variance of the latent space.

    Returns:
        Tensor: Reparameterized samples from the latent space.
    """
    std = torch.exp(0.5 * z_logvar)
    eps = torch.randn_like(std)  # sample from N(0, 1) with the same shape as std
    return z_mean + std * eps


class VAE(AbstractModel):
    """Variational Autoencoder with modular components."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.logger = LogManager.get_logger(__name__)
        self.covariate_embedding_technique = self.model_components.get(
            "covariate_embedding"
        )
        self.logger.info(f"Initializing VAE using {self.covariate_embedding_technique}")

        self.cov_dim = len(self.properties.dataset.covariates) - len(
            self.properties.dataset.skipped_covariates
        )
        latent_dim = self.model_components.get("latent_dim")

        # Change encoder dims for different covariate techniques
        if self.covariate_embedding_technique in {
            "input_feature",
            "conditional_embedding",
            "encoder_embedding",
        }:
            self.encoder_input_dim = self.input_dim + self.cov_dim
            self.encoder_output_dim = latent_dim
        else:
            self.encoder_input_dim = self.input_dim
            self.encoder_output_dim = latent_dim

        # Change decoder dims for different covariate techniques
        if self.covariate_embedding_technique in {
            "input_feature",
            "conditional_embedding",
        }:
            self.decoder_input_dim = latent_dim
            self.decoder_output_dim = self.output_dim + self.cov_dim
        elif self.covariate_embedding_technique == "decoder_embedding":
            self.decoder_input_dim = latent_dim + self.cov_dim
            self.decoder_output_dim = self.output_dim
        else:
            self.decoder_input_dim = latent_dim
            self.decoder_output_dim = self.output_dim

        # Create encoder
        self.encoder = get_encoder(
            encoder_type=self.model_components.get("encoder"),
            input_dim=self.encoder_input_dim,
            hidden_dims=self.properties.model.hidden_layers,
            latent_dim=self.encoder_output_dim,
        )

        # Create decoder
        self.decoder = get_decoder(
            decoder_type=self.model_components.get("decoder"),
            latent_dim=self.decoder_input_dim,
            hidden_dims=self.properties.model.hidden_layers[::-1],
            output_dim=self.decoder_output_dim,
        )

    def forward(
        self, x: Tensor, covariates: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the VAE model.

        :param x: Input data (shape: [B, data_dim])
        :param covariates: Covariates (shape: [B, cov_dim]) or None
        """

        if self.covariate_embedding_technique == "no_embedding":
            z_mean, z_logvar = self.encoder(x)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)  # shape [B, output_dim]
            return x_recon, z_mean, z_logvar

        if self.covariate_embedding_technique == "input_feature":
            # Concat inputs before the encoder
            encoder_input = torch.cat([x, covariates], dim=1)
            z_mean, z_logvar = self.encoder(encoder_input)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)
            return x_recon, z_mean, z_logvar

        if self.covariate_embedding_technique == "conditional_embedding":
            # Concat inputs before the encoder
            encoder_input = torch.cat([x, covariates], dim=1)
            z_mean, z_logvar = self.encoder(encoder_input)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)
            return x_recon, z_mean, z_logvar

        if self.covariate_embedding_technique == "encoder_embedding":
            # Concat inputs before the encoder
            encoder_input = torch.cat([x, covariates], dim=1)
            z_mean, z_logvar = self.encoder(encoder_input)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)
            return x_recon, z_mean, z_logvar

        if self.covariate_embedding_technique == "decoder_embedding":
            z_mean, z_logvar = self.encoder(x)
            z = reparameterize(z_mean, z_logvar)
            # Decoder sees z + covariates
            decoder_input = torch.cat([z, covariates], dim=1)
            x_recon = self.decoder(decoder_input)
            return x_recon, z_mean, z_logvar

        raise ValueError(
            f"Unknown covariate_embedding technique: {self.covariate_embedding_technique}"
        )
