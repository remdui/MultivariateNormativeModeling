"""Tests for the RMSE metric."""

import pytest
import torch
from torch import Tensor

from analysis.metrics.rmse import compute_rmse


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = Tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = Tensor([[1.2, 1.8], [3.1, 3.9]])
    return original, reconstructed


def test_rmse_per_sample(data_tensors):
    """Test RMSE per sample."""
    original, reconstructed = data_tensors
    rmse = compute_rmse(original, reconstructed, metric_type="sample")
    # Expected RMSE per sample: [sqrt((0.2^2 + 0.2^2)/2), sqrt((0.1^2 + 0.1^2)/2)] = [0.2, 0.1]
    expected_rmse = Tensor([0.2, 0.1])
    assert torch.allclose(rmse, expected_rmse), f"Expected {expected_rmse}, got {rmse}"


def test_rmse_per_feature(data_tensors):
    """Test RMSE per feature."""
    original, reconstructed = data_tensors
    rmse = compute_rmse(original, reconstructed, metric_type="feature")
    # Expected RMSE per feature: [sqrt((0.2^2 + 0.1^2)/2), sqrt((0.2^2 + 0.1^2)/2)] = [0.158113883, 0.158113883]
    expected_rmse = Tensor([0.158113883, 0.158113883])
    assert torch.allclose(rmse, expected_rmse), f"Expected {expected_rmse}, got {rmse}"


def test_rmse_total(data_tensors):
    """Test total RMSE."""
    original, reconstructed = data_tensors
    rmse = compute_rmse(original, reconstructed, metric_type="total")
    # Expected total RMSE: sqrt((0.2^2 + 0.2^2 + 0.1^2 + 0.1^2) / 4) = 0.158113883
    expected_rmse = Tensor([0.158113883])
    assert torch.allclose(rmse, expected_rmse), f"Expected {expected_rmse}, got {rmse}"


def test_invalid_rmse_type(data_tensors):
    """Test invalid RMSE type."""
    original, reconstructed = data_tensors
    with pytest.raises(ValueError, match="Invalid rmse_type: invalid. Choose from"):
        compute_rmse(original, reconstructed, metric_type="invalid")


def test_shape_mismatch():
    """Test shape mismatch between original and reconstructed tensors."""
    original = Tensor([[1.0, 2.0]])
    reconstructed = Tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_rmse(original, reconstructed)
