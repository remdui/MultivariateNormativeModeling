"""Variational Autoencoder with modular components."""

import torch
from torch import Tensor

from model.components.factory import get_decoder, get_encoder
from model.models.abstract_model import AbstractModel


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Reparameterize the latent space."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


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
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
