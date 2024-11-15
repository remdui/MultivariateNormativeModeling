"""Jensen-Shannon Divergence (JSD) metric for distributional divergence."""

import torch
from torch import Tensor


def compute_js_divergence(
    original: Tensor,
    reconstructed: Tensor,
    assumption: str = "gaussian",
    epsilon: float = 1e-6,
) -> Tensor:
    """
    Compute Jensen-Shannon Divergence (JSD) per feature, assuming Gaussian distributions.

    JS divergence is a symmetric and smoothed version of the Kullback-Leibler (KL) divergence, measuring the similarity between two probability distributions.
    Note: the discrete version is the same as scipy.spatial.distance.jensenshannon distance squared.

    Equation: JSD = 0.5 * (KL(P || M) + KL(Q || M)), where M = 0.5 * (P + Q)

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        assumption (str): Assumption for the distribution. Choose from 'discrete' or 'gaussian'. Default is 'gaussian'.
        epsilon (float): Small value to avoid division by zero. Default is 1e-6.

    Returns:
        Tensor: JSD for each feature, indicating distributional divergence.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    if assumption == "discrete":
        # Compute the probability distribution of the original and reconstructed data
        # Normalize each feature to create probability distributions
        original_prob = original / (original.sum(dim=0, keepdim=True) + epsilon)
        reconstructed_prob = reconstructed / (
            reconstructed.sum(dim=0, keepdim=True) + epsilon
        )

        # Compute the mean distribution
        mean_prob = (original_prob + reconstructed_prob) / 2

        # Compute KL divergence between each distribution and the mean distribution
        kl_orig_to_mean = torch.sum(
            original_prob
            * (torch.log(original_prob + epsilon) - torch.log(mean_prob + epsilon)),
            dim=0,
        )
        kl_recon_to_mean = torch.sum(
            reconstructed_prob
            * (
                torch.log(reconstructed_prob + epsilon) - torch.log(mean_prob + epsilon)
            ),
            dim=0,
        )

        # JSD is the average of these two KL divergences
        js_divergence = (kl_orig_to_mean + kl_recon_to_mean) / 2

    elif assumption == "gaussian":
        # Calculate the mean and standard deviation of each feature in the original and reconstructed tensors
        mu_p = original.mean(dim=0)  # Mean per feature for original data
        sigma_p = (
            original.std(dim=0) + epsilon
        )  # Std per feature for original data, with epsilon to avoid division by zero

        mu_q = reconstructed.mean(dim=0)  # Mean per feature for reconstructed data
        sigma_q = (
            reconstructed.std(dim=0) + epsilon
        )  # Std per feature for reconstructed data, with epsilon to avoid division by zero

        # Mean and variance of the midpoint distribution per feature
        mu_m = 0.5 * (mu_p + mu_q)
        sigma_m = torch.sqrt(0.5 * (sigma_p**2 + sigma_q**2))

        # Compute KLD(p || m) per feature
        kld_p_m = (
            torch.log(sigma_m / sigma_p)
            + (sigma_p**2 + (mu_p - mu_m) ** 2) / (2 * sigma_m**2)
            - 0.5
        )

        # Compute KLD(q || m) per feature
        kld_q_m = (
            torch.log(sigma_m / sigma_q)
            + (sigma_q**2 + (mu_q - mu_m) ** 2) / (2 * sigma_m**2)
            - 0.5
        )

        # JSD per feature is the average of these two KLDs
        js_divergence = 0.5 * (kld_p_m + kld_q_m)

    else:
        raise ValueError(
            f"Invalid assumption: {assumption}. Choose from 'discrete', 'gaussian'."
        )

    return js_divergence
