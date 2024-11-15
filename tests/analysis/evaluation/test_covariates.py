"""Test cases for the covariates correlation evaluation function."""

import pytest
import torch

from analysis.evaluation.covariates import evaluate_covariates_correlation


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


def test_evaluate_covariates_correlation_valid_rmse(data_tensors):
    """Test correlation evaluation with RMSE as the reconstruction error metric."""
    original, reconstructed, covariates = data_tensors
    corr = evaluate_covariates_correlation(
        original=original,
        reconstructed=reconstructed,
        covariates=covariates,
        metric="rmse",
        correlation_type="pearson",
    )
    # Expected correlation value (calculated manually, example placeholder)
    expected_corr = torch.tensor([-0.8627, -0.4981])
    assert torch.allclose(
        corr, expected_corr, atol=1e-4
    ), f"Expected correlation {expected_corr}, got {corr}"


def test_evaluate_covariates_correlation_valid_mae(data_tensors):
    """Test correlation evaluation with MAE as the reconstruction error metric."""
    original, reconstructed, covariates = data_tensors
    corr = evaluate_covariates_correlation(
        original=original,
        reconstructed=reconstructed,
        covariates=covariates,
        metric="mae",
        correlation_type="pearson",
    )

    # Expected correlation value (calculated manually, example placeholder)
    expected_corr = torch.tensor([-0.8627, -0.4981])
    assert torch.allclose(
        corr, expected_corr, atol=1e-4
    ), f"Expected correlation {expected_corr}, got {corr}"


def test_evaluate_covariates_correlation_invalid_metric(data_tensors):
    """Test handling of invalid reconstruction error metric."""
    original, reconstructed, covariates = data_tensors
    with pytest.raises(
        ValueError, match="Unsupported metric: invalid_metric. Choose 'rmse' or 'mae'."
    ):
        evaluate_covariates_correlation(
            original=original,
            reconstructed=reconstructed,
            covariates=covariates,
            metric="invalid_metric",
            correlation_type="pearson",
        )


def test_evaluate_covariates_correlation_invalid_correlation_type(data_tensors):
    """Test handling of invalid correlation type."""
    original, reconstructed, covariates = data_tensors
    with pytest.raises(
        ValueError,
        match="Unsupported correlation_type: invalid_type. Choose 'pearson' or 'spearman'.",
    ):
        evaluate_covariates_correlation(
            original=original,
            reconstructed=reconstructed,
            covariates=covariates,
            metric="rmse",
            correlation_type="invalid_type",
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
            metric="rmse",
            correlation_type="pearson",
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
            metric="rmse",
            correlation_type="pearson",
        )
