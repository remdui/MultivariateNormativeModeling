"""MLP Encoder Module."""

from torch import nn

from model.components.encoder import BaseEncoder


class MLPEncoder(BaseEncoder):
    """Multi-Layer Perceptron Encoder."""

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.model = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.model(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
