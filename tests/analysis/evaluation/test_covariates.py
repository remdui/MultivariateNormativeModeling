"""Tests for the evaluation of covariates correlation with reconstruction error."""

import pytest
import torch

from analysis.metrics.misc.covariate_corr import evaluate_covariates_correlation
from analysis.metrics.stats.mae import compute_mae
from analysis.metrics.stats.mse import compute_mse


@pytest.fixture(name="data_tensors")
def data_tensors_fixture():
    """Fixture to provide sample tensors for testing."""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    reconstructed = torch.tensor(
        [[1.1, 2.1], [2.9, 4.1], [5.1, 5.9]], dtype=torch.float32
    )

    # Covariates: age and sex (binary: 0 for male, 1 for female)
    covariates = torch.tensor(
        [[25, 0], [35, 1], [45, 0]], dtype=torch.float32
    )  # Columns: age, sex
    return original, reconstructed, covariates


def test_evaluate_covariates_correlation_valid_rse(data_tensors):
    """Test correlation evaluation with MSE as the reconstruction error metric."""
    original, reconstructed, covariates = data_tensors
    corr = evaluate_covariates_correlation(
        original=original,
        reconstructed=reconstructed,
        covariates=covariates,
        metric_fn=compute_mse,
    )
    # Expected correlation value (calculated manually, example placeholder)
    expected_corr = torch.tensor([-0.8660, -0.5000])
    assert torch.allclose(
        corr, expected_corr, atol=1e-3
    ), f"Expected correlation {expected_corr}, got {corr}"


def test_evaluate_covariates_correlation_valid_mae(data_tensors):
    """Test correlation evaluation with MAE as the reconstruction error metric."""
    original, reconstructed, covariates = data_tensors
    corr = evaluate_covariates_correlation(
        original=original,
        reconstructed=reconstructed,
        covariates=covariates,
        metric_fn=compute_mae,
    )

    # Expected correlation value (calculated manually, example placeholder)
    expected_corr = torch.tensor([-0.8627, -0.4981])
    assert torch.allclose(
        corr, expected_corr, atol=1e-3
    ), f"Expected correlation {expected_corr}, got {corr}"


def test_evaluate_covariates_correlation_unsupported_metric(data_tensors):
    """Test handling of unsupported metric functions."""

    def unsupported_metric(original, reconstructed):
        return torch.abs(original - reconstructed).mean(dim=1)

    original, reconstructed, covariates = data_tensors
    with pytest.raises(
        ValueError, match="The provided metric function does not support 'metric_type'."
    ):
        evaluate_covariates_correlation(
            original=original,
            reconstructed=reconstructed,
            covariates=covariates,
            metric_fn=unsupported_metric,
        )


def test_evaluate_covariates_correlation_shape_mismatch_covariates(data_tensors):
    """Test handling of covariate shape mismatch."""
    original, reconstructed, _ = data_tensors
    covariates = torch.tensor(
        [25, 35], dtype=torch.float32
    )  # Mismatched number of samples
    with pytest.raises(
        ValueError, match="Number of covariates must match the number of samples."
    ):
        evaluate_covariates_correlation(
            original=original,
            reconstructed=reconstructed,
            covariates=covariates,
            metric_fn=compute_mse,
        )


def test_evaluate_covariates_correlation_shape_mismatch_reconstruction(data_tensors):
    """Test handling of shape mismatch between original and reconstructed tensors."""
    original, _, covariates = data_tensors
    reconstructed = torch.tensor(
        [[1.1, 2.1], [2.9, 4.1]], dtype=torch.float32
    )  # Mismatched shape
    with pytest.raises(
        ValueError, match="Original and reconstructed tensors must have the same shape."
    ):
        evaluate_covariates_correlation(
            original=original,
            reconstructed=reconstructed,
            covariates=covariates,
            metric_fn=compute_mse,
        )
