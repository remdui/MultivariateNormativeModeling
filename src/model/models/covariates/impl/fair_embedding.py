"""
Variational Fair Autoencoder covariate embedding strategy.

Encodes both x and sensitive covariates s into z, decodes from (z, s)
to reconstruct x, and applies an MMD penalty to z to encourage invariance
across sensitive groups.
"""

import torch
from torch import Tensor

from model.models.covariates.base_embedding_strategy import BaseEmbeddingStrategy
from model.models.util.vae import reparameterize
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


def _compute_mmd_rbf(x: Tensor, y: Tensor, sigma: float = 1.0) -> Tensor:
    """
    Compute the RBF‐kernel MMD between samples x and y:

      MMD^2(x, y) = E[k(x, x')] + E[k(y, y')] - 2 E[k(x, y')]
    where k(a, b) = exp(-||a - b||^2 / (2 * sigma^2)).
    """
    # x: (n1, d), y: (n2, d)
    n1 = x.size(0)
    n2 = y.size(0)
    if n1 == 0 or n2 == 0:
        return torch.tensor(0.0, device=x.device)

    # Pairwise squared distances
    xx = x.unsqueeze(1)  # (n1, 1, d)
    xx_expand = xx.expand(-1, n1, -1)  # (n1, n1, d)
    xxt = x.unsqueeze(0).expand(n1, -1, -1)  # (n1, n1, d)
    dist_xx = ((xx_expand - xxt) ** 2).sum(dim=2)  # (n1, n1)

    yy = y.unsqueeze(1)
    yy_expand = yy.expand(-1, n2, -1)
    yyt = y.unsqueeze(0).expand(n2, -1, -1)
    dist_yy = ((yy_expand - yyt) ** 2).sum(dim=2)  # (n2, n2)

    xy = x.unsqueeze(1).expand(n1, n2, -1)  # (n1, n2, d)
    yx = y.unsqueeze(0).expand(n1, n2, -1)
    dist_xy = ((xy - yx) ** 2).sum(dim=2)  # (n1, n2)

    k_xx = torch.exp(-dist_xx / (2 * sigma**2))
    k_yy = torch.exp(-dist_yy / (2 * sigma**2))
    k_xy = torch.exp(-dist_xy / (2 * sigma**2))

    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd


class FairEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    Variational Fair Autoencoder (VFAE) embedding strategy.

    - encoder takes [x, s] → (z_mean, z_logvar)
    - sample z = reparameterize(z_mean, z_logvar)
    - decoder takes [z, s] → x_recon
    - During loss, an MMD penalty on z across different sensitive‐attribute groups
      enforces that the latent representation z is invariant to s.
    """

    def __init__(
        self,
        vae: any,
        mmd_lambda: float = 1.0,
        covariate_info: dict = None,
        sigma: float = 1.0,
    ) -> None:
        """
        Args:

            vae: The VAE model instance (with encoder and decoder modules).
            mmd_lambda: Weight for the MMD penalty term.
            covariate_info: Dict with keys:
                - "labels": list of all covariate names (e.g., ["age", "sex_M", "sex_F"])
                - "continuous": list of indices for continuous covariates
                - "categorical": dict mapping a group name (e.g. "sex") to indices of one‐hot columns
            sigma: Bandwidth for the RBF kernel in MMD.
        """
        super().__init__(vae)
        self.mmd_lambda = mmd_lambda
        self.sigma = sigma

        # Store covariate metadata
        self.covariate_info = covariate_info or {"labels": []}
        self.cov_dim = len(self.covariate_info.get("labels", []))
        self.continuous_indices = self.covariate_info.get("continuous", [])
        self.categorical_groups = self.covariate_info.get("categorical", {})

    def configure_dimensions(
        self, input_dim: int, output_dim: int, cov_dim: int, latent_dim: int
    ) -> dict:
        """
        In VFAE, encoder_input_dim = input_dim + cov_dim,.

                         encoder_output_dim = latent_dim
                         decoder_input_dim = latent_dim + cov_dim,
                         decoder_output_dim = output_dim
        """
        return {
            "encoder_input_dim": input_dim + cov_dim,
            "encoder_output_dim": latent_dim,
            "decoder_input_dim": latent_dim + cov_dim,
            "decoder_output_dim": output_dim,
        }

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict:
        """
        Forward pass:
          - require covariates (sensitive attributes)
          - encoder_input = [x, covariates]
          - decode_input = [z, covariates]
        Returns:
          {
            "x_recon": reconstructed x,
            "z_mean": mean of q(z|x,s),
            "z_logvar": logvar of q(z|x,s),
            "z": sampled latent code
          }
        """
        if covariates is None:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                "Covariates must be provided for fair embedding."
            )

        encoder_input = torch.cat([x, covariates], dim=1)
        z_mean, z_logvar = self.vae.encoder(encoder_input)
        z = reparameterize(z_mean, z_logvar)
        decoder_input = torch.cat([z, covariates], dim=1)
        x_recon = self.vae.decoder(decoder_input)

        return {
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
        }
