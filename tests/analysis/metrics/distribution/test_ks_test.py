"""Test Kolmogorov-Smirnov test for distribution similarity."""

import pytest
import torch

from analysis.legacy.metrics.distribution.ks_test import kolmogorov_smirnov_test


def test_distribution_similarity_identical():
    """Test distribution similarity with identical original and reconstructed data."""
    original = torch.randn(1000, 5)
    reconstructed = original.clone()  # Identical to original
    results = kolmogorov_smirnov_test(original, reconstructed)

    for feature, result in results.items():
        assert result[
            "similar_distribution"
        ], f"Expected similar distribution for feature {feature}"


def test_distribution_similarity_noisy():
    """Test distribution similarity with slightly noisy reconstructed data."""
    original = torch.randn(1000, 5)
    reconstructed = original + torch.normal(0, 0.1, size=original.shape)
    results = kolmogorov_smirnov_test(original, reconstructed)

    for feature, result in results.items():
        assert result[
            "similar_distribution"
        ], f"Expected similar distribution for feature {feature}"


def test_distribution_similarity_different():
    """Test distribution similarity with very different reconstructed data."""
    original = torch.normal(2, 3, size=(1000, 5))
    reconstructed = torch.normal(
        0, 1, size=original.shape
    )  # Very different from original
    results = kolmogorov_smirnov_test(original, reconstructed)

    for feature, result in results.items():
        assert not result[
            "similar_distribution"
        ], f"Expected different distribution for feature {feature}"


def test_shape_mismatch():
    """Test distribution similarity with shape mismatch."""
    original = torch.randn(1000, 5)
    reconstructed = torch.randn(1000, 6)  # Different number of features
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        kolmogorov_smirnov_test(original, reconstructed)
