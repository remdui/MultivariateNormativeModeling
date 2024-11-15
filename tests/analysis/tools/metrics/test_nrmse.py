"""Tests for the NRMSE metric."""

import pytest
import torch
from torch import Tensor

from analysis.tools.metrics.nmrse import compute_nrmse


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = Tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = Tensor([[1.2, 1.8], [3.1, 3.9]])
    return original, reconstructed


def test_nrmse_range(data_tensors):
    """Test NRMSE with range normalization."""
    original, reconstructed = data_tensors
    nrmse = compute_nrmse(original, reconstructed, normalization_type="range")
    # RMSE = sqrt((0.2^2 + 0.2^2 + 0.1^2 + 0.1^2) / 4) = 0.158113883
    # Range = 4 - 1 = 3
    # NRMSE (range) = 0.158113883 / 3 = 0.052704628
    expected_nrmse = Tensor([0.052704628])
    assert torch.allclose(
        nrmse, expected_nrmse, atol=1e-6
    ), f"Expected {expected_nrmse}, got {nrmse}"


def test_nrmse_mean(data_tensors):
    """Test NRMSE with mean normalization."""
    original, reconstructed = data_tensors
    nrmse = compute_nrmse(original, reconstructed, normalization_type="mean")
    # RMSE = 0.158113883 (as calculated above)
    # Mean = (1 + 2 + 3 + 4) / 4 = 2.5
    # NRMSE (mean) = 0.158113883 / 2.5 = 0.063245553
    expected_nrmse = Tensor([0.063245553])
    assert torch.allclose(
        nrmse, expected_nrmse, atol=1e-6
    ), f"Expected {expected_nrmse}, got {nrmse}"


def test_invalid_normalization_type(data_tensors):
    """Test invalid normalization type."""
    original, reconstructed = data_tensors
    with pytest.raises(
        ValueError,
        match="Invalid normalization_type: invalid. Choose from 'range' or 'mean'.",
    ):
        compute_nrmse(original, reconstructed, normalization_type="invalid")


def test_zero_range_error():
    """Test error when range of original data is zero."""
    original = Tensor([[1.0, 1.0], [1.0, 1.0]])
    reconstructed = Tensor([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(
        ValueError,
        match="Range of original data is zero; cannot compute NRMSE with 'range' normalization.",
    ):
        compute_nrmse(original, reconstructed, normalization_type="range")


def test_zero_mean_error():
    """Test error when mean of original data is zero."""
    original = Tensor([[0.0, 0.0], [0.0, 0.0]])
    reconstructed = Tensor([[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(
        ValueError,
        match="Mean of original data is zero; cannot compute NRMSE with 'mean' normalization.",
    ):
        compute_nrmse(original, reconstructed, normalization_type="mean")
