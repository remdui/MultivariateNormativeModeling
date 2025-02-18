"""Models for getting prior distributions."""

import torch
from torch import nn


class CovariatePriorNet(nn.Module):
    """Learns p(z | "covariate"]) = N( mu("covariate"), sigma^2("covariate") ).

    "covariate" -> [Hidden Layers] -> prior_mu, prior_logvar
    """

    def __init__(self, latent_dim: int, hidden_dims: list[int], num_covariates: int):
        super().__init__()
        layers = []
        prev_dim = num_covariates
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # The core MLP
        self.shared = nn.Sequential(*layers)

        # Final heads
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, age: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Age: shape [batch_size, 1] (continuous age).

        Returns:
          prior_mu, prior_logvar: each shape [batch_size, latent_dim]
        """
        h = self.shared(age)
        prior_mu = self.fc_mu(h)
        prior_logvar = self.fc_logvar(h)
        return prior_mu, prior_logvar
