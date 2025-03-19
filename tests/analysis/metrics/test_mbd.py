"""Tests for Mean Bias Deviation metric."""

import pytest
import torch

from analysis.metrics.stats.mbd import compute_mbd


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    reconstructed = torch.tensor([[1.1, 2.1], [2.1, 4.1], [3.1, 6.1]])
    return original, reconstructed


def test_mbd_per_feature(data_tensors):
    """Test Mean Bias Deviation per feature with positive bias."""
    original, reconstructed = data_tensors
    mbd = compute_mbd(original, reconstructed)

    # Expected MBD: Average bias per feature
    expected_mbd = torch.tensor([0.1, 0.1])
    assert torch.allclose(
        mbd, expected_mbd, atol=1e-3
    ), f"Expected {expected_mbd}, got {mbd}"


def test_mbd_no_bias():
    """Test Mean Bias Deviation when there is no systematic bias."""
    original = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    reconstructed = original.clone()  # No difference between original and reconstructed
    mbd = compute_mbd(original, reconstructed)

    # Expected MBD: No bias, so should be zero
    expected_mbd = torch.tensor([0.0, 0.0])
    assert torch.allclose(
        mbd, expected_mbd, atol=1e-3
    ), f"Expected {expected_mbd}, got {mbd}"


def test_mbd_negative_bias():
    """Test Mean Bias Deviation with negative bias."""
    original = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    reconstructed = torch.tensor([[0.9, 1.9], [1.9, 3.9], [2.9, 5.9]])
    mbd = compute_mbd(original, reconstructed)

    # Expected MBD: Average negative bias per feature
    expected_mbd = torch.tensor([-0.1, -0.1])
    assert torch.allclose(
        mbd, expected_mbd, atol=1e-3
    ), f"Expected {expected_mbd}, got {mbd}"


def test_shape_mismatch():
    """Test shape mismatch between original and reconstructed tensors."""
    original = torch.tensor([[1.0, 2.0]])
    reconstructed = torch.tensor([[1.0, 2.0, 3.0]])  # Mismatched shape
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_mbd(original, reconstructed)
