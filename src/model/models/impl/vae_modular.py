"""Variational Autoencoder with modular components."""

import torch
from torch import Tensor

from model.components.base_decoder import BaseDecoder
from model.components.base_encoder import BaseEncoder
from model.models.abstract_model import AbstractModel


class VAE(AbstractModel):
    """Variational Autoencoder with modular components."""

    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparametrize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparametrize the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the VAE model."""
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
