"""Tests for the R² Score metric."""

import pytest
import torch

from analysis.metrics.stats.r2 import compute_r2_score


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    reconstructed = torch.tensor([[1.1, 2.1], [2.9, 3.9], [5.1, 5.9]])
    return original, reconstructed


def test_r2_per_feature(data_tensors):
    """Test R² Score per feature."""
    original, reconstructed = data_tensors
    r2 = compute_r2_score(original, reconstructed)
    # Manually calculated expected R² per feature
    expected_r2 = torch.tensor([0.9963, 0.9963])
    assert torch.allclose(
        r2, expected_r2, atol=1e-3
    ), f"Expected {expected_r2}, got {r2}"


def test_zero_variance_feature():
    """Test R² score when a feature in the original data has zero variance."""
    original = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    reconstructed = torch.tensor([[1.1, 2.1], [1.0, 2.0], [1.1, 2.1]])
    r2 = compute_r2_score(original, reconstructed)
    # Expected R² for the first feature with zero variance should be zero
    expected_r2 = torch.tensor([0.0, 0.0])
    assert torch.allclose(
        r2, expected_r2, atol=1e-3
    ), f"Expected {expected_r2}, got {r2}"


def test_shape_mismatch():
    """Test shape mismatch between original and reconstructed tensors."""
    original = torch.tensor([[1.0, 2.0]])
    reconstructed = torch.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_r2_score(original, reconstructed)
