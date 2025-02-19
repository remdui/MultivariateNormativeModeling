"""MLP Decoder Module."""

from torch import Tensor, nn

from model.components.decoder.base_decoder import BaseDecoder


class MLPDecoder(BaseDecoder):
    """
    Multi-Layer Perceptron (MLP) Decoder.

    This decoder transforms latent representations into output data via a series of
    linear layers with activation functions, followed by a final linear layer.
    Optionally, a final activation function can be applied.
    """

    def __init__(
        self, latent_dim: int, hidden_dims: list[int], output_dim: int
    ) -> None:
        """
        Initialize the MLPDecoder.

        Args:
            latent_dim (int): Dimensionality of the latent input.
            hidden_dims (list[int]): List of hidden layer sizes.
            output_dim (int): Dimensionality of the output.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # Optionally, insert a normalization layer here:
            # layers.append(self._get_normalization_layer(h_dim))
            layers.append(self.activation_function)
            prev_dim = h_dim

        self.model = nn.Sequential(*layers)
        self.final_layer = nn.Linear(prev_dim, output_dim)
        # Optionally, assign a final activation function:
        # self.output_activation = self.final_activation_function

    def forward(self, x: Tensor) -> Tensor:
        """
        Execute the forward pass of the decoder.

        Args:
            x (Tensor): Input latent tensor with shape [batch_size, latent_dim].

        Returns:
            Tensor: Reconstructed output tensor with shape [batch_size, output_dim].
        """
        x = self.model(x)
        # Uncomment the following line if a final activation is desired:
        # x = self.output_activation(self.final_layer(x))
        x = self.final_layer(x)
        return x
