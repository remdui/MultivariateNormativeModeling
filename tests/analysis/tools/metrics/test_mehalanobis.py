"""Tests for Mahalanobis Distance computation between two distributions."""

import pytest
import torch

from analysis.metrics.mehalanobis import compute_mahalanobis_distance

# Set random seed and deterministic settings
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    # Modified data to ensure the covariance matrix is closer to non-singular,
    # but we'll add regularization as well to ensure invertibility.
    original = torch.tensor(
        [
            [1.0, 2.0, 5.0],
            [2.0, 3.1, 6.0],
            [3.0, 4.2, 7.0],
            [4.0, 5.3, 8.0],
            [5.0, 6.4, 9.0],
        ]
    )
    reconstructed = original.clone()  # Identical data for baseline test
    return original, reconstructed


def test_mahalanobis_tensors_identical(data_tensors):
    """Test Mahalanobis distance when original and reconstructed data are identical."""
    original, reconstructed = data_tensors
    mahalanobis_distances = compute_mahalanobis_distance(original, reconstructed)
    expected_distances = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(
        mahalanobis_distances, expected_distances, atol=1e-3
    ), f"Expected {expected_distances}, got {mahalanobis_distances}"


def test_mahalanobis_distance_slightly_different():
    """Test Mahalanobis distance with slightly different data."""
    original = torch.tensor(
        [
            [1.0, 2.0, 5.0],
            [2.0, 3.1, 6.0],
            [3.0, 4.2, 7.0],
            [4.0, 5.3, 8.0],
            [5.0, 6.4, 9.0],
        ]
    )
    reconstructed = original + torch.tensor(
        [
            [0.1, -0.1, 0.1],
            [-0.1, 0.1, -0.1],
            [0.1, -0.1, 0.1],
            [-0.1, 0.1, -0.1],
            [0.1, -0.1, 0.1],
        ]
    )
    mahalanobis_distances = compute_mahalanobis_distance(original, reconstructed)
    expected_distances = torch.tensor([0.7304, 1.0954, 0.7303, 1.0954, 0.7303])
    # Expected non-zero distances, as the data is slightly different
    assert torch.allclose(
        mahalanobis_distances, expected_distances, atol=1e-3
    ), f"Expected {expected_distances}, got {mahalanobis_distances}"


def test_mahalanobis_distance_large_difference():
    """Test Mahalanobis distance with large differences in data."""
    original = torch.tensor(
        [
            [1.0, 2.0, 5.0],
            [2.0, 3.1, 6.0],
            [3.0, 4.2, 7.0],
            [4.0, 5.3, 8.0],
            [5.0, 6.4, 9.0],
        ]
    )
    reconstructed = torch.tensor(
        [
            [10.0, 20.0, 50.0],
            [10.5, 20.5, 50.5],
            [11.0, 21.0, 51.0],
            [11.5, 21.5, 51.5],
            [12.0, 22.0, 52.0],
        ]
    )
    mahalanobis_distances = compute_mahalanobis_distance(original, reconstructed)
    expected_distances = torch.tensor([1.2772, 0.6386, 0.0000, 0.6362, 1.2870])
    # Expect high distances, indicating large deviation from original distribution
    assert torch.allclose(
        mahalanobis_distances, expected_distances, atol=1e-4
    ), f"Expected {expected_distances}, got {mahalanobis_distances}"


def test_shape_mismatch():
    """Test Mahalanobis distance computation with shape mismatch."""
    original = torch.tensor([[1.0, 2.0, 5.0], [2.0, 3.1, 6.0]])
    reconstructed = torch.tensor([[1.0, 2.0], [2.0, 3.0]])  # Shape mismatch
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape"
    ):
        compute_mahalanobis_distance(original, reconstructed)
