"""Kolmogorov-Smirnov (KS) test to compare the distribution of original and reconstructed data."""

from scipy.stats import ks_2samp  # type: ignore
from torch import Tensor


def kolmogorov_smirnov_test(
    original: Tensor, reconstructed: Tensor, alpha: float = 0.05
) -> dict:
    """
    Perform a Kolmogorov-Smirnov (KS) test per feature to compare the distribution.

    of the original and reconstructed data.

    KS test is a non-parametric test that compares the cumulative distribution.

    Args:
        original (Tensor): Original data tensor (num_samples, num_features).
        reconstructed (Tensor): Reconstructed data tensor (num_samples, num_features).
        alpha (float): Significance level for the KS test. Default is 0.05.

    Returns:
        dict: Dictionary with p-values per feature and a conclusion on whether
              the distributions are significantly different.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed tensors must have the same shape.")

    # Dictionary to store p-values and significance conclusion per feature
    distribution_similarity = {}

    # Perform KS test per feature
    for feature in range(original.shape[1]):
        original_feature = original[:, feature].numpy()
        reconstructed_feature = reconstructed[:, feature].numpy()

        # Perform KS test
        _, p_value = ks_2samp(original_feature, reconstructed_feature)

        # Check if p-value is above the significance level
        similar_distribution = p_value > alpha
        distribution_similarity[feature] = {
            "p_value": p_value,
            "similar_distribution": similar_distribution,
        }

    return distribution_similarity
