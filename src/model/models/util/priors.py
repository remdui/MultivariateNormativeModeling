"""Models for computing conditional prior distributions.

This module defines the CovariatePriorNet class, which learns a conditional prior
distribution p(z | covariate) modeled as a Gaussian with parameters
mu(covariate) and log(sigma²(covariate)). The network maps input covariate(s)
through an MLP to produce the mean and log-variance of the Gaussian prior.
"""

import torch
from torch import nn


class CovariatePriorNet(nn.Module):
    """
    Neural network for learning a conditional prior distribution.

    Given input covariate(s) of dimension `num_covariates`, this network processes the
    input through a series of fully-connected hidden layers (with ReLU activations) and outputs
    the mean and log-variance for a Gaussian prior over latent variables of dimension `latent_dim`.
    """

    def __init__(self, latent_dim: int, hidden_dims: list[int], num_covariates: int):
        """
        Initialize the CovariatePriorNet.

        Args:
            latent_dim (int): Dimensionality of the latent space (output size).
            hidden_dims (list[int]): List of hidden layer sizes for the MLP.
            num_covariates (int): Number of input covariates.
        """
        super().__init__()
        layers = []
        prev_dim = num_covariates
        # Build the MLP with hidden layers.
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.shared = nn.Sequential(*layers)  # Core MLP for feature extraction.
        # Two output heads: one for the mean and one for the log-variance.
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, covariates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the conditional prior parameters from the input covariates.

        Args:
            covariates (torch.Tensor): Input tensor of covariates with shape [batch_size, num_covariates].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple (prior_mu, prior_logvar), each of shape
            [batch_size, latent_dim], representing the mean and log-variance of the Gaussian prior.
        """
        h = self.shared(covariates)
        prior_mu = self.fc_mu(h)
        prior_logvar = self.fc_logvar(h)
        return prior_mu, prior_logvar
