"""MLP Encoder Module."""

from torch import Tensor, nn

from model.components.encoder.base_encoder import BaseEncoder


class MLPEncoder(BaseEncoder):
    """Multi-Layer Perceptron Encoder."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        """
        Initialize the MLPEncoder.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dims (list[int]): List specifying the number of neurons in each hidden layer.
            latent_dim (int): Dimensionality of the latent space.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, h_dim))

            # Append normalization layer if available
            norm_layer = self.get_normalization_layer(h_dim)
            if norm_layer is not None:
                layers.append(norm_layer)

            # Append activation function (unique instance)
            layers.append(self.get_activation_function())

            # Append dropout (regularization) if available
            reg_layer = self.get_regularization()
            if reg_layer is not None:
                layers.append(reg_layer)

            prev_dim = h_dim

        self.model = nn.Sequential(*layers)
        # Two parallel heads for computing latent mean and log-variance.
        self.fc_z_mean = nn.Linear(prev_dim, latent_dim)
        self.fc_z_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the forward pass of the encoder.

        Args:
            x (Tensor): Input tensor with shape [batch_size, input_dim].

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - z_mean: Mean of the latent Gaussian distribution.
                - z_logvar: Log variance of the latent Gaussian distribution.
        """
        x = self.model(x)
        z_mean = self.fc_z_mean(x)
        z_logvar = self.fc_z_logvar(x)
        return z_mean, z_logvar
