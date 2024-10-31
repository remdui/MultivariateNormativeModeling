"""MLP Decoder Module."""

from torch import Tensor, nn

from model.components.decoder import BaseDecoder


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
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.model = nn.Sequential(*layers)
        self.final_layer = nn.Linear(prev_dim, output_dim)
        self.output_activation = nn.Sigmoid()  # Change if needed

    def forward(self, z: Tensor) -> Tensor:
        h = self.model(z)
        x_recon = self.output_activation(self.final_layer(h))
        return x_recon
