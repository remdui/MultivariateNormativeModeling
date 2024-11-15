"""Tests for Jensen-Shannon Divergence (JSD) computation between two distributions."""

import pytest
import torch

from analysis.metrics.distribution.js_div import compute_js_divergence


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]]
    )
    reconstructed = original.clone()  # Identical data for baseline test
    return original, reconstructed


def test_js_divergence_identical(data_tensors):
    """Test JSD when original and reconstructed data are identical."""
    original, reconstructed = data_tensors
    js_div = compute_js_divergence(original, reconstructed)
    expected_js = torch.zeros_like(js_div)
    assert torch.allclose(
        js_div, expected_js, atol=1e-3
    ), f"Expected {expected_js}, got {js_div}"


def test_js_divergence_slightly_different():
    """Test JSD with slightly different mean and variance."""
    original = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]]
    )
    reconstructed = original + torch.tensor(
        [[0.1, -0.1], [-0.1, 0.1], [0.1, -0.1], [-0.1, 0.1], [0.1, -0.1]]
    )
    js_div = compute_js_divergence(original, reconstructed)
    expected_js = torch.tensor(
        [0.0001, 0.0001]
    )  # Expected values may vary depending on data
    assert torch.allclose(
        js_div, expected_js, atol=1e-3
    ), f"Expected {expected_js}, got {js_div}"


def test_js_divergence_large_difference():
    """Test JSD with large differences in mean and variance."""
    original = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]]
    )
    reconstructed = torch.tensor(
        [[10.0, 20.0], [10.5, 20.5], [11.0, 21.0], [11.5, 21.5], [12.0, 22.0]]
    )
    js_div = compute_js_divergence(original, reconstructed)
    expected_js = torch.tensor(
        [16.2, 64.8]
    )  # Expected values may vary depending on data
    assert torch.allclose(
        js_div, expected_js, atol=1e-3
    ), f"Expected {expected_js}, got {js_div}"


def test_shape_mismatch():
    """Test JSD with shape mismatch between original and reconstructed tensors."""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = torch.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_js_divergence(original, reconstructed)
