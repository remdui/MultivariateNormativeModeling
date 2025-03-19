"""Tests for the Pearson’s r metric."""

import pytest
import torch

from analysis.legacy.metrics.stats.pearson_r import compute_pearson_r


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    reconstructed = torch.tensor([[0.9, 2.1], [2.9, 4.1], [5.1, 5.9]])
    return original, reconstructed


def test_pearson_r_per_feature(data_tensors):
    """Test Pearson’s r per feature with typical data."""
    original, reconstructed = data_tensors
    pearson_r = compute_pearson_r(original, reconstructed)

    # Manually computed expected Pearson's r values for each feature
    expected_pearson_r = torch.tensor(
        [0.9995, 0.9995]
    )  # Example values, adjust if needed
    assert torch.allclose(
        pearson_r, expected_pearson_r, atol=1e-3
    ), f"Expected {expected_pearson_r}, got {pearson_r}"


def test_pearson_r_with_zero_variance_feature():
    """Test Pearson’s r when one feature has zero variance in the original or reconstructed data."""
    original = torch.tensor(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    )  # Zero variance in first feature
    reconstructed = torch.tensor([[1.1, 2.1], [1.0, 2.0], [0.9, 1.9]])
    pearson_r = compute_pearson_r(original, reconstructed)

    # Expected Pearson's r: First feature should be 0 due to zero variance
    expected_pearson_r = torch.tensor([0.0, 0.0])
    assert torch.allclose(
        pearson_r, expected_pearson_r, atol=1e-3
    ), f"Expected {expected_pearson_r}, got {pearson_r}"


def test_shape_mismatch():
    """Test shape mismatch between original and reconstructed tensors."""
    original = torch.tensor([[1.0, 2.0]])
    reconstructed = torch.tensor([[1.0, 2.0, 3.0]])  # Mismatched shape
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_pearson_r(original, reconstructed)


def test_perfect_positive_correlation():
    """Test Pearson’s r for perfectly correlated data."""
    original = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    reconstructed = original * 2  # Perfect positive correlation
    pearson_r = compute_pearson_r(original, reconstructed)

    # Expected Pearson's r: 1.0 for perfectly correlated data
    expected_pearson_r = torch.tensor([1.0, 1.0])
    assert torch.allclose(
        pearson_r, expected_pearson_r, atol=1e-3
    ), f"Expected {expected_pearson_r}, got {pearson_r}"


def test_perfect_negative_correlation():
    """Test Pearson’s r for perfectly negatively correlated data."""
    original = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    reconstructed = original * -1  # Perfect negative correlation
    pearson_r = compute_pearson_r(original, reconstructed)

    # Expected Pearson's r: -1.0 for perfectly negatively correlated data
    expected_pearson_r = torch.tensor([-1.0, -1.0])
    assert torch.allclose(
        pearson_r, expected_pearson_r, atol=1e-3
    ), f"Expected {expected_pearson_r}, got {pearson_r}"


def test_nontrivial_correlation():
    """Test Pearson’s r with nontrivial positive correlation in both features."""
    original = torch.tensor(
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]
    )
    # Introduce a slight, consistent difference to create moderate positive correlation
    reconstructed = torch.tensor(
        [[1.2, 1.9], [2.1, 4.1], [3.2, 5.8], [4.1, 8.1], [4.9, 10.2]]
    )
    pearson_r = compute_pearson_r(original, reconstructed)

    # Expected Pearson's r values should be moderately high but not perfect
    # These are approximate expected values based on the moderate correlation setup
    expected_pearson_r = torch.tensor([0.9986, 0.9992])
    assert torch.allclose(
        pearson_r, expected_pearson_r, atol=1e-3
    ), f"Expected {expected_pearson_r}, got {pearson_r}"
