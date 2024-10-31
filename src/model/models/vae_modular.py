"""Variational Autoencoder with modular components."""

import torch
from torch import Tensor, nn

from model.components.decoder import BaseDecoder
from model.components.encoder import BaseEncoder


class VAE(nn.Module):
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
