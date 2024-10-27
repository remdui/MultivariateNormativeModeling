"""Basic Variational Autoencoder (VAE) model with covariates embedded as input nodes."""

import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    """Basic Variational Autoencoder (VAE) model with covariates embedded as input nodes."""

    def __init__(self, input_dim, hidden_dim, latent_dim, covariate_dim):
        """Constructor for the VAE class."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim + covariate_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance
        self.fc3 = nn.Linear(latent_dim + covariate_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, covariates):
        """Encode the input data and covariates into the latent space."""
        x = torch.cat([x, covariates], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        """Reparametrize the latent space.

        Reparameterization trick explanation: https://towardsdatascience.com/reparameterization-trick-126062cfd3c3
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, covariates):
        """Decode the latent space and covariates into the output."""
        z = torch.cat([z, covariates], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, covariates):
        """Forward pass of the VAE model."""
        mu, logvar = self.encode(x, covariates)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, covariates), mu, logvar
