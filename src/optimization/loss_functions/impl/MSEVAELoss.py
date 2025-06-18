"""Custom loss function for VAE with beta-VAE and KL annealing (MSE variant).

This module defines the MSEVAELoss class, which computes the loss for a variational autoencoder (VAE)
by combining a Mean Squared Error (MSE) reconstruction term with a KL divergence term. The KL weight is
annealed over training epochs using a KLAnnealing schedule. Covariate embedding options are supported.
"""

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss

from model.models.covariates.factory import get_embedding_technique
from model.models.covariates.impl.fair_embedding import _compute_mmd_rbf
from optimization.loss_functions.util.kl_annealing import KLAnnealing
from util.errors import UnsupportedCovariateEmbeddingTechniqueError


def _rbf_kernel(x: Tensor, sigma: float) -> Tensor:
    """
    Computes the RBF kernel matrix K for input x:

      K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))
    """
    n, d = x.shape
    # Compute pairwise squared Euclidean distances
    # Expand to (n, n, d) to compute (x_i - x_j)^2 along dim=2
    xx = x.unsqueeze(1).expand(n, n, d)
    yy = x.unsqueeze(0).expand(n, n, d)
    dist2 = ((xx - yy) ** 2).sum(dim=2)  # shape (n, n)
    return torch.exp(-dist2 / (2 * sigma**2))


def _center_kernel(K: Tensor) -> Tensor:
    """
    Centers a kernel matrix K in feature space via H = I - (1/n) 11^T:

      K_centered = H K H
    """
    n = K.size(0)
    # Create centering matrix H = I - (1/n) * 11^T
    identity = torch.eye(n, device=K.device, dtype=K.dtype)
    ones = torch.ones((n, n), device=K.device, dtype=K.dtype) / n
    H = identity - ones
    return H @ K @ H


def _compute_hsic(
    z: Tensor,
    s: Tensor,
    sigma_z: float,
    sigma_s: float,
) -> Tensor:
    """
    Computes the (biased) empirical HSIC between z and s:

      HSIC = (1 / n^2) * trace( K_centered(z) @ K_centered(s) )
    where K(z) and K(s) are RBF kernels.
    """
    n = z.size(0)
    if n < 2:
        # If batch size is 1, HSIC is zero by definition
        return torch.tensor(0.0, device=z.device)

    # Compute RBF kernels
    K_z = _rbf_kernel(z, sigma_z)  # (n, n)
    K_s = _rbf_kernel(s, sigma_s)  # (n, n)

    # Center both
    Kc_z = _center_kernel(K_z)
    Kc_s = _center_kernel(K_s)

    # Biased HSIC estimate
    hsic_val = (Kc_z * Kc_s).sum() / (n * n)
    return hsic_val


class MSEVAELoss(_WeightedLoss):
    """
    MSE + KL Divergence loss for VAE with beta-VAE and KL annealing.

    Computes reconstruction loss using MSE and adds a KL divergence penalty weighted by an annealed beta.
    Depending on the covariate embedding technique, the reconstruction loss may be computed directly or
    after processing covariate data.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: Any = None,
        reduce: Any = None,
        reduction: str = "mean",
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        kl_anneal_start: int = 0,
        kl_anneal_end: int = 0,
        mmd_sigma: float = 1.0,
        mmd_lambda: float = 1.0,
        hsic_sigma_z: float = 1.0,
        hsic_sigma_s: float = 1.0,
        hsic_lambda: float = 1.0,
    ) -> None:
        """
        Initialize the MSEVAELoss.

        Args:
            weight, size_average, reduce, reduction: Inherited from _WeightedLoss.
            beta_start (float): Initial KL weight.
            beta_end (float): Final KL weight.
            kl_anneal_start (int): Epoch to begin annealing.
            kl_anneal_end (int): Epoch to finish annealing.
        """
        super().__init__(weight, size_average, reduce, reduction)
        self.kl_annealer = KLAnnealing(
            beta_start=beta_start,
            beta_end=beta_end,
            kl_anneal_start=kl_anneal_start,
            kl_anneal_end=kl_anneal_end,
        )
        self.covariate_embedding_technique = get_embedding_technique()
        self.mmd_sigma = mmd_sigma
        self.mmd_lambda = mmd_lambda
        self.hsic_sigma_z = hsic_sigma_z
        self.hsic_sigma_s = hsic_sigma_s
        self.hsic_lambda = hsic_lambda

    def forward(
        self,
        model_outputs: dict[str, Tensor],
        x: Tensor,
        covariates: Tensor | None = None,
        covariate_labels: list[str] | None = None,
        current_epoch: int | None = None,
    ) -> Tensor:
        """
        Compute the VAE loss as the sum of MSE reconstruction loss and a weighted KL divergence term.

        Args:
            model_outputs (dict[str, Tensor]): Dictionary from the model's forward pass containing:
                - "x_recon": The reconstructed input.
                - "z_mean": The posterior mean.
                - "z_logvar": The posterior log-variance.
            x (Tensor): Original input tensor.
            covariates (Tensor | None, optional): Covariate data required for certain embedding modes.
            covariate_labels (list[str] | None, optional): Covariate labels for logging.
            current_epoch (int | None, optional): Current epoch for KL annealing; if None, final beta is used.

        Returns:
            Tensor: A scalar loss value.
        """
        recon_x = model_outputs["x_recon"]
        z_mean = model_outputs.get("z_mean")
        z_logvar = model_outputs.get("z_logvar")

        cov_dim = len(covariate_labels)

        beta = (
            self.kl_annealer.compute_beta(current_epoch)
            if current_epoch is not None
            else self.kl_annealer.beta_end
        )

        _, data_dim = x.shape

        if self.covariate_embedding_technique in {
            "no_embedding",
            "encoder_embedding",
            "decoder_embedding",
            "encoderdecoder_embedding",
        }:
            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)
            mse_cov = 0.0

        elif self.covariate_embedding_technique == "bag_encoderdecoder_embedding":
            # ── Include both z and g in the KL term ──────────────────────────
            # Extract posterior means and log-vars for z and g
            z_mean = model_outputs["z_mean"]
            z_logvar = model_outputs["z_logvar"]
            g_mean = model_outputs["g_mean"]
            g_logvar = model_outputs["g_logvar"]
            # Concatenate so the later KL computation covers both blocks
            z_mean = torch.cat([z_mean, g_mean], dim=1)
            z_logvar = torch.cat([z_logvar, g_logvar], dim=1)

            # Reconstruction loss
            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)
            mse_cov = 0.0

            # Optional HSIC penalty to enforce z ⟂ age
            z = model_outputs.get("z")
            if z is not None and covariates is not None:
                age = covariates[:, 0:1]
                hsic_pen = _compute_hsic(
                    z, age, sigma_z=self.hsic_sigma_z, sigma_s=self.hsic_sigma_s
                )
                mse_data = mse_data + self.hsic_lambda * hsic_pen

        elif self.covariate_embedding_technique == "input_feature_embedding":
            if covariates is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Covariates must be provided for 'input_feature' mode."
                )
            x_combined = torch.cat([x, covariates], dim=1)
            mse_data = F.mse_loss(recon_x, x_combined, reduction=self.reduction)
            mse_cov = 0.0
        elif self.covariate_embedding_technique == "conditional_embedding":
            if covariates is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Covariates must be provided for 'conditional_embedding' mode."
                )
            recon_data = recon_x[:, :data_dim]
            recon_cov = recon_x[:, data_dim : data_dim + cov_dim]
            mse_data = F.mse_loss(recon_data, x, reduction=self.reduction)

            continuous_indices: Any = []
            categorical_groups: Any = {}
            for i, label in enumerate(covariate_labels):
                if "_" not in label:
                    continuous_indices.append(i)
                else:
                    group = label.split("_")[0]
                    categorical_groups.setdefault(group, []).append(i)

            loss_cont = 0.0
            if continuous_indices:
                pred_cont = recon_cov[:, continuous_indices]
                true_cont = covariates[:, continuous_indices]
                loss_cont = F.mse_loss(pred_cont, true_cont, reduction=self.reduction)

            loss_cat = 0.0
            for group, indices in categorical_groups.items():
                pred_cat = recon_cov[:, indices]
                true_onehot = covariates[:, indices]
                true_labels = true_onehot.argmax(dim=1)
                loss_cat += F.cross_entropy(
                    pred_cat, true_labels, reduction=self.reduction
                )

            mse_cov = loss_cont + loss_cat * 1.0

        elif self.covariate_embedding_technique in {
            "adversarial_embedding",
            "conditional_adversarial_embedding",
        }:
            if covariates is None or covariate_labels is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Covariates and covariate_labels must be provided for adversarial mode."
                )

            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)
            mse_cov = 0.0

            adv_preds = model_outputs.get("adv_preds", {})
            continuous_indices: list[int] = []
            categorical_groups: dict[str, list[int]] = {}

            for i, label in enumerate(covariate_labels):
                if "_" not in label:
                    continuous_indices.append(i)
                else:
                    grp = label.split("_")[0]
                    categorical_groups.setdefault(grp, []).append(i)

            loss_adv_cont = 0.0
            if continuous_indices and "continuous" in adv_preds:
                pred_cont = adv_preds["continuous"]
                true_cont = covariates[:, continuous_indices]

                alpha = 0.1  # ← scale factor
                loss_adv_cont = alpha * F.mse_loss(
                    pred_cont, true_cont, reduction=self.reduction
                )

            loss_adv_cat = 0.0
            for grp, indices in categorical_groups.items():
                if grp in adv_preds:
                    pred_cat = adv_preds[grp]
                    true_onehot = covariates[:, indices]
                    true_labels = true_onehot.argmax(dim=1)
                    loss_adv_cat += F.cross_entropy(
                        pred_cat, true_labels, reduction=self.reduction
                    )

            adv_loss = loss_adv_cont + loss_adv_cat
            mse_cov += adv_loss

        elif self.covariate_embedding_technique == "fair_embedding":
            if covariates is None or covariate_labels is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Covariates and covariate_labels must be provided for 'fair_embedding' mode."
                )

            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)
            mse_cov = 0.0

            z = model_outputs.get("z")
            if z is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Latent code 'z' missing in model outputs for 'fair_embedding'."
                )

            categorical_groups: dict[str, list[int]] = {}
            for i, label in enumerate(covariate_labels):
                if "_" not in label:
                    continue
                grp = label.split("_")[0]
                categorical_groups.setdefault(grp, []).append(i)

            mmd_total = torch.tensor(0.0, device=z.device)

            # print(covariates) # Find out order (same as config order)

            for group, indices in categorical_groups.items():
                # `indices` is a list of one‐hot column positions for this group,
                # e.g. [2,3,4] if "site_A","site_B","site_C" are at columns 2,3,4.
                num_classes = len(indices)
                if num_classes < 2:
                    # trivial or degenerate group → skip
                    continue

                # Build true_labels ∈ {0,…, num_classes−1}
                true_onehot = covariates[:, indices]  # shape (batch, num_classes)
                true_labels = true_onehot.argmax(dim=1)  # shape (batch,)

                # For each distinct pair (i, j) with 0 ≤ i < j < num_classes,
                #     collect z_i and z_j, then compute MMD(z_i, z_j).
                group_mmd = torch.tensor(0.0, device=z.device)
                pair_count = 0

                for i_class in range(num_classes):
                    # z_i = all z where true_labels == i_class
                    z_i = z[true_labels == i_class]
                    # skip if no samples of that class in this batch
                    if z_i.size(0) == 0:
                        continue

                    for j_class in range(i_class + 1, num_classes):
                        z_j = z[true_labels == j_class]
                        if z_j.size(0) == 0:
                            continue

                        # compute MMD between z_i and z_j
                        mmd_ij = _compute_mmd_rbf(z_i, z_j, sigma=self.mmd_sigma)
                        group_mmd = group_mmd + mmd_ij
                        pair_count += 1

                if pair_count > 0:
                    # Option A: sum all pairwise MMDs
                    mmd_total = mmd_total + group_mmd

                    # Option B (alternative): average over pairs:
                    # mmd_total = mmd_total + (group_mmd / float(pair_count))

            mse_data = mse_data + self.mmd_lambda * mmd_total

        # In MSEVAELoss.forward(...)
        elif self.covariate_embedding_technique == "bag_fair_embedding":
            # 1) KL on both z & g
            z_mean = model_outputs["z_mean"]
            z_logvar = model_outputs["z_logvar"]
            g_mean = model_outputs["g_mean"]
            g_logvar = model_outputs["g_logvar"]
            z_mean = torch.cat([z_mean, g_mean], dim=1)
            z_logvar = torch.cat([z_logvar, g_logvar], dim=1)

            # 2) reconstruction loss
            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)
            mse_cov = 0.0

            # 3) MMD penalty on z to enforce fairness
            z = model_outputs.get("z")
            if z is None or covariates is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "z and covariates required for bag_fair_embedding."
                )
            categorical = {}
            for i, label in enumerate(covariate_labels):
                if "_" not in label:
                    continue
                grp = label.split("_")[0]
                categorical.setdefault(grp, []).append(i)

            mmd_total = torch.tensor(0.0, device=z.device)
            for grp, indices in categorical.items():
                true_onehot = covariates[:, indices]
                labels = true_onehot.argmax(dim=1)
                classes = torch.unique(labels)
                for i in range(len(classes)):
                    for j in range(i + 1, len(classes)):
                        zi = z[labels == classes[i]]
                        zj = z[labels == classes[j]]
                        mmd_total += _compute_mmd_rbf(zi, zj, sigma=self.mmd_sigma)
            mse_data += self.mmd_lambda * mmd_total

        elif self.covariate_embedding_technique == "hsic_embedding":
            if covariates is None or covariate_labels is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Covariates and covariate_labels must be provided for 'hsic_embedding'."
                )
            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)
            mse_cov = 0.0

            z = model_outputs.get("z")
            if z is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Latent code 'z' missing for HSIC embedding."
                )

            hsic_val = _compute_hsic(
                z, covariates, sigma_z=self.hsic_sigma_z, sigma_s=self.hsic_sigma_s
            )
            extra_penalty = self.hsic_lambda * hsic_val
            mse_data += extra_penalty

        elif self.covariate_embedding_technique == "disentangle_embedding":
            if covariates is None or covariate_labels is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Covariates and covariate_labels must be provided for 'disentangle_embedding'."
                )

            mse_data = F.mse_loss(recon_x, x, reduction=self.reduction)

            disc_preds = model_outputs.get("disc_preds", {})
            continuous_indices: list[int] = []
            categorical_groups: dict[str, list[int]] = {}

            for i, label in enumerate(covariate_labels):
                if "_" not in label:
                    continuous_indices.append(i)
                else:
                    grp = label.split("_")[0]
                    categorical_groups.setdefault(grp, []).append(i)

            loss_disc_cont = 0.0
            if continuous_indices and "continuous" in disc_preds:
                pred_cont = disc_preds["continuous"]
                true_cont = covariates[:, continuous_indices]
                loss_disc_cont = F.mse_loss(
                    pred_cont, true_cont, reduction=self.reduction
                )

            loss_disc_cat = 0.0
            for grp, indices in categorical_groups.items():
                if grp in disc_preds:
                    pred_cat = disc_preds[grp]  # shape = (batch, num_classes)
                    true_onehot = covariates[:, indices]
                    true_labels = true_onehot.argmax(dim=1)
                    loss_disc_cat += F.cross_entropy(
                        pred_cat, true_labels, reduction=self.reduction
                    )

            disc_loss = loss_disc_cont + loss_disc_cat

            z = model_outputs.get("z")
            if z is None:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    "Latent code 'z' missing for disentangle embedding."
                )

            cov_dim = len(continuous_indices) + sum(
                len(indices) for indices in categorical_groups.values()
            )
            if z.size(1) < cov_dim:
                raise UnsupportedCovariateEmbeddingTechniqueError(
                    f"Latent dimension {z.size(1)} smaller than required sensitive dimension {cov_dim}."
                )
            z_u = z[:, cov_dim:]

            hsic_val = _compute_hsic(
                z_u, covariates, sigma_z=self.hsic_sigma_z, sigma_s=self.hsic_sigma_s
            )
            hsic_penalty = self.hsic_lambda * hsic_val
            mse_cov = disc_loss + hsic_penalty

        else:
            raise UnsupportedCovariateEmbeddingTechniqueError(
                f"Unknown covariate embedding technique: {self.covariate_embedding_technique}"
            )

        recon_loss = mse_data + mse_cov

        if self.reduction == "mean":
            kld_per_sample = -0.5 * torch.sum(
                1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1
            )
            kld = kld_per_sample.mean()
        else:
            kld = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        return recon_loss + beta * kld
