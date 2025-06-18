"""BAG fair embedding."""

import torch
from torch import Tensor

from model.models.covariates.base_embedding_strategy import BaseEmbeddingStrategy
from model.models.util.vae import reparameterize
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


def _compute_mmd_rbf(x: Tensor, y: Tensor, sigma: float = 1.0) -> Tensor:
    # same as in FairEmbeddingStrategy…
    n1, n2 = x.size(0), y.size(0)
    if n1 == 0 or n2 == 0:
        return torch.tensor(0.0, device=x.device)
    xx = x.unsqueeze(1).expand(n1, n1, -1)
    xxt = x.unsqueeze(0).expand(n1, n1, -1)
    yy = y.unsqueeze(1).expand(n2, n2, -1)
    yyt = y.unsqueeze(0).expand(n2, n2, -1)
    dist_xx = ((xx - xxt) ** 2).sum(-1)
    dist_yy = ((yy - yyt) ** 2).sum(-1)
    xy = x.unsqueeze(1).expand(n1, n2, -1)
    yx = y.unsqueeze(0).expand(n1, n2, -1)
    dist_xy = ((xy - yx) ** 2).sum(-1)
    k_xx = torch.exp(-dist_xx / (2 * sigma**2))
    k_yy = torch.exp(-dist_yy / (2 * sigma**2))
    k_xy = torch.exp(-dist_xy / (2 * sigma**2))
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


class BagFairEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    BAG + Fair (MMD) embedding:

      - encoder → [z_mean,g_mean], [z_logvar,g_logvar]
      - z: age-invariant block, g: brain-age-gap block
      - decoder conditions on brain_age = age + g
      - MMD penalty on z across sensitive groups enforced in the loss
    """

    def __init__(
        self,
        vae,
        gap_dim: int = 1,
        age_index: int = 0,
        covariate_info: dict | None = None,
        mmd_lambda: float = 1.0,
        sigma: float = 1.0,
    ):
        super().__init__(vae)
        self.gap_dim = gap_dim
        self.age_index = age_index
        self.cov_info = covariate_info or {"labels": [], "categorical": {}}
        self.mmd_lambda = mmd_lambda
        self.sigma = sigma
        self.latent_size = None

    def configure_dimensions(
        self,
        input_dim: int,
        output_dim: int,
        cov_dim: int,
        latent_dim: int,
    ) -> dict:
        self.latent_size = latent_dim
        return {
            "encoder_input_dim": input_dim + cov_dim,
            "encoder_output_dim": latent_dim + self.gap_dim,
            "decoder_input_dim": latent_dim + cov_dim,
            "decoder_output_dim": output_dim,
        }

    def forward(self, x: Tensor, covariates: Tensor | None) -> dict:
        if covariates is None:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                "Covariates required for BagFairEmbedding."
            )

        # encode x and all covs
        enc_in = torch.cat([x, covariates], dim=1)
        h_mean, h_logvar = self.vae.encoder(enc_in)

        # split into z and g
        z_mean, g_mean = torch.split(h_mean, [self.latent_size, self.gap_dim], dim=1)
        z_logvar, g_logvar = torch.split(
            h_logvar, [self.latent_size, self.gap_dim], dim=1
        )

        z = reparameterize(z_mean, z_logvar)
        g = reparameterize(g_mean, g_logvar)

        # compute brain age
        age = covariates[:, self.age_index : self.age_index + 1]
        brain_age = age + g

        # replace age in covariates for decoder
        if self.age_index == 0:
            cov_dec = torch.cat([brain_age, covariates[:, 1:]], dim=1)
        else:
            cov_dec = torch.cat(
                [
                    covariates[:, : self.age_index],
                    brain_age,
                    covariates[:, self.age_index + 1 :],
                ],
                dim=1,
            )

        # decode
        dec_in = torch.cat([z, cov_dec], dim=1)
        x_recon = self.vae.decoder(dec_in)

        return {
            "x_recon": x_recon,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "g_mean": g_mean,
            "g_logvar": g_logvar,
            "z": z,
            "g": g,
            "brain_age": brain_age,
        }
