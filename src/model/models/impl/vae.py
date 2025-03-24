"""Variational Autoencoder with modular components.

This module implements a Variational Autoencoder (VAE) that supports multiple covariate
embedding techniques and conditional priors. The VAE is built using modular encoder and
decoder components, and employs the reparameterization trick for sampling from the latent space.
"""

from torch import Tensor

from entities.log_manager import LogManager
from model.components.factory import get_decoder, get_encoder
from model.models.abstract_model import AbstractModel
from model.models.covariates.factory import get_embedding_strategy
from model.models.util.covariates import get_enabled_covariate_count
from model.models.util.priors import CovariatePriorNet


class VAE(AbstractModel):
    """
    Variational Autoencoder with modular components.

    This VAE supports various covariate embedding techniques and conditional priors.
    The encoder and decoder dimensions are configured by the embedding strategy.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the VAE model.

        Args:
            input_dim (int): Dimension of the input data.
            output_dim (int): Dimension of the output data.
        """
        super().__init__(input_dim, output_dim)
        self.logger = LogManager.get_logger(__name__)
        self.embedding_strategy = get_embedding_strategy(self)
        cov_dim = get_enabled_covariate_count()
        latent_dim = self.model_components.get("latent_dim")

        # Delegate dimension configuration to the strategy.
        dims = self.embedding_strategy.configure_dimensions(
            self.input_dim, self.output_dim, cov_dim, latent_dim
        )
        self.encoder_input_dim = dims["encoder_input_dim"]
        self.encoder_output_dim = dims["encoder_output_dim"]
        self.decoder_input_dim = dims["decoder_input_dim"]
        self.decoder_output_dim = dims["decoder_output_dim"]

        # Age-conditional prior network (used for age_prior_embedding).
        self.age_prior_net = CovariatePriorNet(latent_dim, [32, 16], num_covariates=1)

        # Create encoder and decoder using factory methods.
        self.encoder = get_encoder(
            encoder_type=self.model_components.get("encoder"),
            input_dim=self.encoder_input_dim,
            hidden_dims=self.properties.model.hidden_layers,
            latent_dim=self.encoder_output_dim,
        )
        self.decoder = get_decoder(
            decoder_type=self.model_components.get("decoder"),
            latent_dim=self.decoder_input_dim,
            hidden_dims=self.properties.model.hidden_layers[::-1],
            output_dim=self.decoder_output_dim,
        )

    def forward(self, x: Tensor, covariates: Tensor | None = None) -> dict:
        """
        Execute the forward pass of the VAE by delegating to the embedding strategy.

        Args:
            x (Tensor): Input data of shape [B, data_dim].
            covariates (Tensor | None): Optional covariate data of shape [B, cov_dim].

        Returns:
            dict[str, Tensor]: Dictionary containing VAE outputs.
        """
        return self.embedding_strategy.forward(x, covariates)
