"""Tests for the SSIM metric."""

import pytest
import torch
from torch import Tensor

from analysis.legacy.metrics.image.ssim import compute_ssim


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = Tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = Tensor([[1.1, 2.1], [2.9, 3.9]])
    return original, reconstructed


def test_ssim_identical(data_tensors):
    """Test SSIM when original and reconstructed data are identical."""
    original, _ = data_tensors
    ssim = compute_ssim(original, original)
    expected_ssim = torch.ones_like(ssim)  # SSIM should be 1 for identical data
    assert torch.allclose(
        ssim, expected_ssim, atol=1e-6
    ), f"Expected {expected_ssim}, got {ssim}"


def test_ssim_similar_data(data_tensors):
    """Test SSIM for similar but not identical data."""
    original, reconstructed = data_tensors
    ssim = compute_ssim(original, reconstructed)
    expected_ssim = torch.tensor([0.9945, 0.9945])
    assert torch.allclose(
        ssim, expected_ssim, atol=1e-4
    ), f"Expected {expected_ssim}, got {ssim}"


def test_ssim_different_data():
    """Test SSIM for completely different data."""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    reconstructed = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
    ssim = compute_ssim(original, reconstructed)
    expected_ssim = torch.tensor([0.0392, 0.0392])
    assert torch.allclose(
        ssim, expected_ssim, atol=1e-4
    ), f"Expected {expected_ssim}, got {ssim}"


def test_shape_mismatch():
    """Test SSIM with shape mismatch between original and reconstructed tensors."""
    original = Tensor([[1.0, 2.0]])
    reconstructed = Tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_ssim(original, reconstructed)
