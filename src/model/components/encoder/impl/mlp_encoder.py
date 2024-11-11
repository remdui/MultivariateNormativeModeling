"""MLP Encoder Module."""

from torch import Tensor, nn

from model.components.encoder.base_encoder import BaseEncoder


class MLPEncoder(BaseEncoder):
    """Multi-Layer Perceptron Encoder."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self.activation_function)
            prev_dim = h_dim
        self.model = nn.Sequential(*layers)
        self.fc_z_mean = nn.Linear(prev_dim, latent_dim)
        self.fc_z_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.model(x)
        z_mean = self.fc_z_mean(x)
        z_logvar = self.fc_z_logvar(x)
        return z_mean, z_logvar
