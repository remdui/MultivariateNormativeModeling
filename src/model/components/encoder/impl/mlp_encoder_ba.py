"""MLP Encoder Module."""

from torch import Tensor, nn

from model.components.encoder.base_encoder import BaseEncoder


class MLPEncoderBA(BaseEncoder):
    """
    Multi-Layer Perceptron (MLP) Encoder with an auxiliary branch.

    This encoder maps input features through a series of linear layers with activation functions,
    producing parameters for two distributions:
      - (z_mean, z_logvar) for the primary latent space.
      - (ba_mean, ba_logvar) for an auxiliary branch.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        """
        Initialize the MLPEncoderBA.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dims (list[int]): Sizes of the hidden layers.
            latent_dim (int): Dimensionality of the primary latent space.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # Optionally, insert a normalization layer here.
            layers.append(self.activation_function)
            prev_dim = h_dim

        self.model = nn.Sequential(*layers)
        # Heads for primary latent space.
        self.fc_z_mean = nn.Linear(prev_dim, latent_dim)
        self.fc_z_logvar = nn.Linear(prev_dim, latent_dim)
        # Heads for auxiliary branch.
        self.fc_ba_mean = nn.Linear(prev_dim, 1)
        self.fc_ba_logvar = nn.Linear(prev_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform the forward pass.

        Processes the input through the MLP, then computes:
            - z_mean and z_logvar for the primary latent variables.
            - ba_mean and ba_logvar for the auxiliary branch.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                - z_mean: Mean of the primary latent distribution.
                - z_logvar: Log variance of the primary latent distribution.
                - ba_mean: Mean of the auxiliary branch.
                - ba_logvar: Log variance of the auxiliary branch.
        """
        x = self.model(x)
        z_mean = self.fc_z_mean(x)
        z_logvar = self.fc_z_logvar(x)
        ba_mean = self.fc_ba_mean(x)
        ba_logvar = self.fc_ba_logvar(x)
        return z_mean, z_logvar, ba_mean, ba_logvar
