"""Tests for the MAE metric."""

import pytest
import torch
from torch import Tensor

from analysis.metrics.stats.mae import compute_mae


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = Tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = Tensor([[1.2, 1.8], [3.1, 3.9]])
    return original, reconstructed


def test_mae_per_sample(data_tensors):
    """Test MAE per sample."""
    original, reconstructed = data_tensors
    mae = compute_mae(original, reconstructed, metric_type="sample")
    # Expected MAE per sample: [(|1.0 - 1.2| + |2.0 - 1.8|)/2, (|3.0 - 3.1| + |4.0 - 3.9|)/2] = [0.2, 0.1]
    expected_mae = Tensor([0.2, 0.1])
    assert torch.allclose(mae, expected_mae), f"Expected {expected_mae}, got {mae}"


def test_mae_per_feature(data_tensors):
    """Test MAE per feature."""
    original, reconstructed = data_tensors
    mae = compute_mae(original, reconstructed, metric_type="feature")
    # Expected MAE per feature: [(|1.0 - 1.2| + |3.0 - 3.1|)/2, (|2.0 - 1.8| + |4.0 - 3.9|)/2] = [0.15, 0.15]
    expected_mae = Tensor([0.15, 0.15])
    assert torch.allclose(mae, expected_mae), f"Expected {expected_mae}, got {mae}"


def test_mae_total(data_tensors):
    """Test total MAE."""
    original, reconstructed = data_tensors
    mae = compute_mae(original, reconstructed, metric_type="total")
    # Expected total MAE: (|1.0 - 1.2| + |2.0 - 1.8| + |3.0 - 3.1| + |4.0 - 3.9|) / 4 = 0.15
    expected_mae = Tensor([0.15])
    assert torch.allclose(mae, expected_mae), f"Expected {expected_mae}, got {mae}"


def test_invalid_mae_type(data_tensors):
    """Test invalid MAE type."""
    original, reconstructed = data_tensors
    with pytest.raises(ValueError, match="Invalid mae_type: invalid. Choose from"):
        compute_mae(original, reconstructed, metric_type="invalid")


def test_shape_mismatch():
    """Test shape mismatch between original and reconstructed tensors."""
    original = Tensor([[1.0, 2.0]])
    reconstructed = Tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_mae(original, reconstructed)
