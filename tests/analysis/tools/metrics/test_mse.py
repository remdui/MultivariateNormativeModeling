"""Tests for the MSE metric."""

import pytest
import torch
from torch import Tensor

from analysis.tools.metrics.mse import compute_mse


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = Tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = Tensor([[1.2, 1.8], [3.1, 3.9]])
    return original, reconstructed


def test_mse_per_sample(data_tensors):
    """Test MSE per sample."""
    original, reconstructed = data_tensors
    mse = compute_mse(original, reconstructed, mse_type="sample")
    # Expected MSE per sample: [(0.2^2 + 0.2^2)/2, (0.1^2 + 0.1^2)/2] = [0.04, 0.01]
    expected_mse = Tensor([0.04, 0.01])
    assert torch.allclose(mse, expected_mse), f"Expected {expected_mse}, got {mse}"


def test_mse_per_feature(data_tensors):
    """Test MSE per feature."""
    original, reconstructed = data_tensors
    mse = compute_mse(original, reconstructed, mse_type="feature")
    # Expected MSE per feature: [(0.2^2 + 0.1^2)/2, (0.2^2 + 0.1^2)/2] = [0.025, 0.025]
    expected_mse = Tensor([0.025, 0.025])
    assert torch.allclose(mse, expected_mse), f"Expected {expected_mse}, got {mse}"


def test_mse_total(data_tensors):
    """Test total MSE."""
    original, reconstructed = data_tensors
    mse = compute_mse(original, reconstructed, mse_type="total")
    # Expected total MSE: (0.2^2 + 0.2^2 + 0.1^2 + 0.1^2) / 4 = 0.025
    expected_mse = Tensor([0.025])
    assert torch.allclose(mse, expected_mse), f"Expected {expected_mse}, got {mse}"


def test_invalid_mse_type(data_tensors):
    """Test invalid MSE type."""
    original, reconstructed = data_tensors
    with pytest.raises(ValueError, match="Invalid mse_type: invalid. Choose from"):
        compute_mse(original, reconstructed, mse_type="invalid")


def test_shape_mismatch():
    """Test shape mismatch between original and reconstructed tensors."""
    original = Tensor([[1.0, 2.0]])
    reconstructed = Tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_mse(original, reconstructed)
