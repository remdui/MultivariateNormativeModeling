"""Variational Autoencoder with modular components."""

import torch
from torch import Tensor

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

        self.encoder = get_encoder(
            encoder_type=self.model_components.get("encoder"),
            input_dim=self.input_dim,
            hidden_dims=self.properties.model.hidden_layers,
            latent_dim=self.model_components.get("latent_dim"),
        )

        self.decoder = get_decoder(
            decoder_type=self.model_components.get("decoder"),
            latent_dim=self.model_components.get("latent_dim"),
            hidden_dims=self.properties.model.hidden_layers[::-1],
            output_dim=self.output_dim,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the VAE model."""
        z_mean, z_logvar = self.encoder(x)
        z = reparameterize(z_mean, z_logvar)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_logvar
