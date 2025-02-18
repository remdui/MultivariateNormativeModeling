"""Variational Autoencoder with modular components."""

import torch
from torch import Tensor

from entities.log_manager import LogManager
from model.components.factory import get_decoder, get_encoder
from model.models.abstract_model import AbstractModel
from model.models.util.priors import CovariatePriorNet


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
            # "age_prior_embedding"
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
            # "age_prior_embedding"
        }:
            self.decoder_input_dim = latent_dim
            self.decoder_output_dim = self.output_dim + self.cov_dim
        elif self.covariate_embedding_technique == "decoder_embedding":
            self.decoder_input_dim = latent_dim + self.cov_dim
            self.decoder_output_dim = self.output_dim
        else:
            self.decoder_input_dim = latent_dim
            self.decoder_output_dim = self.output_dim

        # 1) Age‐conditional prior
        self.age_prior_net = CovariatePriorNet(latent_dim, [32, 16], num_covariates=1)

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

    def forward(self, x: Tensor, covariates: Tensor | None = None) -> dict[str, Tensor]:
        """
        Forward pass of the VAE model, returning a dictionary of outputs.

        Args:
            x (Tensor): Input data of shape [B, data_dim].
            covariates (Tensor | None): Covariates of shape [B, cov_dim], or None.

        Returns:
            dict[str, Tensor]: A dictionary with keys:
                - "x_recon": Reconstructed output (always present).
                - "z_mean", "z_logvar": Posterior mean/logvar of shape [B, latent_dim].
                - "z": Reparameterized sample from q(z|x, ...).
                - "prior_mu", "prior_logvar": (Optional) If using age_prior_embedding.
                  The learned conditional prior parameters of shape [B, latent_dim].
        """
        outputs = {}

        if self.covariate_embedding_technique == "age_prior_embedding":
            # x_cat = torch.cat([x, covariates], dim=1)
            z_mean, z_logvar = self.encoder(x)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)
            prior_mu, prior_logvar = self.age_prior_net(covariates)

            outputs["x_recon"] = x_recon
            outputs["z_mean"] = z_mean
            outputs["z_logvar"] = z_logvar
            outputs["z"] = z
            outputs["prior_mu"] = prior_mu
            outputs["prior_logvar"] = prior_logvar

        elif self.covariate_embedding_technique == "no_embedding":
            z_mean, z_logvar = self.encoder(x)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)

            outputs["x_recon"] = x_recon
            outputs["z_mean"] = z_mean
            outputs["z_logvar"] = z_logvar
            outputs["z"] = z

        elif self.covariate_embedding_technique == "input_feature":
            encoder_input = torch.cat([x, covariates], dim=1)
            z_mean, z_logvar = self.encoder(encoder_input)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)

            outputs["x_recon"] = x_recon
            outputs["z_mean"] = z_mean
            outputs["z_logvar"] = z_logvar
            outputs["z"] = z

        elif self.covariate_embedding_technique == "conditional_embedding":
            encoder_input = torch.cat([x, covariates], dim=1)
            z_mean, z_logvar = self.encoder(encoder_input)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)

            outputs["x_recon"] = x_recon
            outputs["z_mean"] = z_mean
            outputs["z_logvar"] = z_logvar
            outputs["z"] = z

        elif self.covariate_embedding_technique == "encoder_embedding":
            encoder_input = torch.cat([x, covariates], dim=1)
            z_mean, z_logvar = self.encoder(encoder_input)
            z = reparameterize(z_mean, z_logvar)
            x_recon = self.decoder(z)

            outputs["x_recon"] = x_recon
            outputs["z_mean"] = z_mean
            outputs["z_logvar"] = z_logvar
            outputs["z"] = z

        elif self.covariate_embedding_technique == "decoder_embedding":
            z_mean, z_logvar = self.encoder(x)
            z = reparameterize(z_mean, z_logvar)
            decoder_input = torch.cat([z, covariates], dim=1)
            x_recon = self.decoder(decoder_input)

            outputs["x_recon"] = x_recon
            outputs["z_mean"] = z_mean
            outputs["z_logvar"] = z_logvar
            outputs["z"] = z

        else:
            raise ValueError(
                f"Unknown covariate_embedding technique: {self.covariate_embedding_technique}"
            )

        return outputs
