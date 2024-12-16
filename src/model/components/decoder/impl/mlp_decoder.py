"""MLP Decoder Module."""

from torch import Tensor, nn

from model.components.decoder.base_decoder import BaseDecoder


class MLPDecoder(BaseDecoder):
    """Multi-Layer Perceptron Decoder."""

    def __init__(
        self, latent_dim: int, hidden_dims: list[int], output_dim: int
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = latent_dim
        for _, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self._get_normalization_layer(h_dim))
            layers.append(self.activation_function)
            prev_dim = h_dim
        self.model = nn.Sequential(*layers)
        self.final_layer = nn.Linear(prev_dim, output_dim)
        self.output_activation = self.final_activation_function

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        # x = self.output_activation(self.final_layer(x))
        x = self.final_layer(
            x
        )  # If we have z-scored the data, we don't need to apply the activation function
        return x
