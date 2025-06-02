"""Adversarial VAE covariate embedding strategy."""

import torch
from torch import Tensor, nn

from entities.properties import Properties
from model.models.covariates.base_embedding_strategy import BaseEmbeddingStrategy
from model.models.util.vae import reparameterize
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


class GradientReversalFunction(torch.autograd.Function):
    """Implements a gradient reversal layer (Ganin et al., 2016)."""

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # Reverse gradients and scale by lambda_
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wraps the GRL function into a Module."""

    def __init__(self, lambda_adv: float = 0.1):
        super().__init__()
        self.lambda_adv = lambda_adv

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return GradientReversalFunction.apply(x, self.lambda_adv)


class SimpleConditionalAdversarialEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    “Stabilized” Adversarial embedding:

      - Encoder:    [x] → (z_mean, z_logvar)
      - Reparameterize: z = reparameterize(z_mean, z_logvar)
      - Decoder:    [z] → x_recon
      - Adversary:  GRL(z_mean) → small MLP predicting covariates

    The encoder learns to “fool” the adversary by removing covariate info from z_mean.
    """

    def __init__(
        self,
        vae: any,
        lambda_adv: float = 0.1,
        covariate_info: dict | None = None,
        hidden_dim: int = 64,
        dropout_p: float = 0.1,
    ) -> None:
        """
        Args:

          vae: The VAE instance (must have .encoder and .decoder).
          lambda_adv: GRL weight (e.g. 0.1). Lower → more stable.
          covariate_info: {
              "labels": [...],            # list of cov names (e.g. ["age","sex_M","sex_F"])
              "continuous": [i1, i2, ...],# indices in cov vector that are continuous
              "categorical": {
                   group_name: [i_j, i_k, ...],  # one‐hot columns for group
              }
          }
          hidden_dim: hidden size of adversary MLP (default: 64)
          dropout_p: dropout probability in adversary (default: 0.1)
        """
        super().__init__(vae)
        self.lambda_adv = lambda_adv
        self.grl = GradientReversalLayer(lambda_adv)

        self.covariate_info = covariate_info or {"labels": []}
        self.continuous_indices = self.covariate_info.get("continuous", [])
        self.categorical_groups = self.covariate_info.get("categorical", {})

        self.adversary_cont: nn.Module | None = None
        self.adversary_cat: nn.ModuleDict | None = None

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

    def configure_dimensions(
        self, input_dim: int, output_dim: int, cov_dim: int, latent_dim: int
    ) -> dict:
        """
        - encoder_input_dim  = input_dim (no covariate conditioning here).

        - encoder_output_dim = latent_dim
        - decoder_input_dim  = latent_dim
        - decoder_output_dim = output_dim

        Then build adversary MLPs that take z_mean (latent_dim) and predict covariates.
        We place them on Properties.system.device.
        """
        properties = Properties.get_instance()
        device = properties.system.device

        if self.continuous_indices:
            self.adversary_cont = nn.Sequential(
                nn.Linear(latent_dim, self.hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.hidden_dim, len(self.continuous_indices)),
            ).to(device)
        else:
            self.adversary_cont = None

        # --- Categorical adversary heads (one per group) ---
        self.adversary_cat = nn.ModuleDict()
        for group_name, indices in self.categorical_groups.items():
            num_classes = len(indices)
            self.adversary_cat[group_name] = nn.Sequential(
                nn.Linear(latent_dim, self.hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.hidden_dim, num_classes),
            ).to(device)

        return {
            "encoder_input_dim": input_dim + cov_dim,
            "encoder_output_dim": latent_dim,
            "decoder_input_dim": latent_dim + cov_dim,
            "decoder_output_dim": output_dim,
        }

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict:
        """
        Args:

          x: Tensor of shape (batch_size, input_dim)
          covariates: Tensor of shape (batch_size, cov_dim); must not be None here.

        Returns:
          {
            "x_recon": <Tensor>(batch_size, output_dim),
            "z_mean":   <Tensor>(batch_size, latent_dim),
            "z_logvar": <Tensor>(batch_size, latent_dim),
            "z":        <Tensor>(batch_size, latent_dim),
            "adv_preds": {
                "continuous": <Tensor>(batch_size, #continuous) if any,
                "<group_name>": <Tensor>(batch_size, #classes) for each categorical group
            }
          }
        """
        if covariates is None:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                "Covariates must be provided for stabilized adversarial embedding."
            )

        encoder_input = torch.cat([x, covariates], dim=1)
        z_mean, z_logvar = self.vae.encoder(encoder_input)
        z = reparameterize(z_mean, z_logvar)
        decoder_input = torch.cat([z, covariates], dim=1)
        x_recon = self.vae.decoder(decoder_input)
        z_for_adv = self.grl(z_mean)
        adv_outputs: dict[str, Tensor] = {}

        if self.adversary_cont is not None:
            adv_outputs["continuous"] = self.adversary_cont(z_for_adv)

        for group_name, head in self.adversary_cat.items():
            adv_outputs[group_name] = head(z_for_adv)

        return {
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
            "adv_preds": adv_outputs,
        }
