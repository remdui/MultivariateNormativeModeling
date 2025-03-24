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
            # Linear layer for this hidden dimension
            layers.append(nn.Linear(prev_dim, h_dim))

            # Append normalization layer if available (each call gets a unique instance)
            norm_layer = self.get_normalization_layer(h_dim)
            if norm_layer is not None:
                layers.append(norm_layer)

            # Append activation function (each call gets a unique instance)
            layers.append(self.get_activation_function())

            # Append dropout (regularization) if available
            reg_layer = self.get_regularization()
            if reg_layer is not None:
                layers.append(reg_layer)

            prev_dim = h_dim

        self.model = nn.Sequential(*layers)
        self.final_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Execute the forward pass of the decoder.

        Args:
            x (Tensor): Input latent tensor with shape [batch_size, latent_dim].

        Returns:
            Tensor: Reconstructed output tensor with shape [batch_size, output_dim].
        """
        x = self.model(x)
        x = self.final_layer(x)
        return x
