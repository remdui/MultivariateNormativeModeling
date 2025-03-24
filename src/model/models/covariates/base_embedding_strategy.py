"""Base class for covariate modeling strategies."""

from typing import Any

from torch import Tensor


class BaseEmbeddingStrategy:
    """Base class for covariate modeling strategies."""

    def __init__(self, vae: Any) -> None:
        """
        Args:

            vae: Reference to the VAE instance to access shared components (encoder, decoder, etc.)
        """
        self.vae = vae

    def configure_dimensions(
        self, input_dim: int, output_dim: int, cov_dim: int, latent_dim: int
    ) -> dict[str, int]:
        """
        Configure and return the dimensions for the encoder and decoder.

        Returns a dictionary with:
            - encoder_input_dim
            - encoder_output_dim
            - decoder_input_dim
            - decoder_output_dim
        """
        raise NotImplementedError("Subclasses must implement configure_dimensions.")

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict[str, Tensor]:
        """
        Perform the forward pass using the specific embedding technique.

        Returns a dictionary containing outputs such as the reconstruction,
        latent means/variances, etc.
        """
        raise NotImplementedError("Subclasses must implement forward.")
