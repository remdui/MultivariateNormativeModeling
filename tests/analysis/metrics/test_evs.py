"""Tests for Explained Variance Score (EVS) metric."""

import pytest
import torch

from analysis.legacy.metrics.stats.evs import compute_evs


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    reconstructed = torch.tensor([[1.1, 2.1], [2.9, 3.9], [5.1, 5.9]])
    return original, reconstructed


def test_explained_variance_per_feature(data_tensors):
    """Test Explained Variance Score per feature."""
    original, reconstructed = data_tensors
    evs = compute_evs(original, reconstructed)
    # Manually calculated expected EVS per feature
    expected_evs = torch.tensor([0.99666667, 0.99666667])
    assert torch.allclose(
        evs, expected_evs, atol=1e-3
    ), f"Expected {expected_evs}, got {evs}"


def test_zero_variance_feature():
    """Test EVS when a feature in the original data has zero variance."""
    original = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    reconstructed = torch.tensor([[1.1, 2.1], [1.0, 2.0], [1.1, 2.1]])
    evs = compute_evs(original, reconstructed)
    # Expected EVS for the first feature with zero variance should be zero
    expected_evs = torch.tensor([0.0, 0.0])
    assert torch.allclose(
        evs, expected_evs, atol=1e-3
    ), f"Expected {expected_evs}, got {evs}"


def test_shape_mismatch():
    """Test shape mismatch between original and reconstructed tensors."""
    original = torch.tensor([[1.0, 2.0]])
    reconstructed = torch.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_evs(original, reconstructed)
