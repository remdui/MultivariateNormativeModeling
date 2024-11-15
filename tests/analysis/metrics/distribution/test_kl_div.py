"""Tests for KL divergence computation between two distributions."""

import pytest
import torch

from analysis.metrics.distribution.kl_div import compute_kl_divergence


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]]
    )
    reconstructed = original.clone()  # Identical data for baseline test
    return original, reconstructed


def test_kl_divergence_identical(data_tensors):
    """Test KL divergence when original and reconstructed data are identical."""
    original, reconstructed = data_tensors
    kl_div = compute_kl_divergence(original, reconstructed)
    expected_kl = torch.zeros_like(kl_div)
    assert torch.allclose(
        kl_div, expected_kl, atol=1e-3
    ), f"Expected {expected_kl}, got {kl_div}"


def test_kl_divergence_slightly_different():
    """Test KL divergence with slightly different mean and variance."""
    original = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]]
    )
    reconstructed = original + torch.tensor(
        [[0.1, -0.1], [-0.1, 0.1], [0.1, -0.1], [-0.1, 0.1], [0.1, -0.1]]
    )
    kl_div = compute_kl_divergence(original, reconstructed)
    expected_kl = torch.tensor([0.0005, 0.0005])
    assert torch.allclose(
        kl_div, expected_kl, atol=1e-3
    ), f"Expected {expected_kl}, got {kl_div}"


def test_kl_divergence_large_difference():
    """Test KL divergence with large differences in mean and variance."""
    original = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]]
    )
    reconstructed = torch.tensor(
        [[10.0, 20.0], [10.5, 20.5], [11.0, 21.0], [11.5, 21.5], [12.0, 22.0]]
    )
    kl_div = compute_kl_divergence(original, reconstructed)
    expected_kl = torch.tensor([81.0, 324.0])
    assert torch.allclose(
        kl_div, expected_kl, atol=1e-3
    ), f"Expected {expected_kl}, got {kl_div}"


def test_shape_mismatch():
    """Test KL divergence with shape mismatch between original and reconstructed tensors."""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = torch.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_kl_divergence(original, reconstructed)
