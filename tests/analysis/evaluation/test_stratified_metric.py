"""Tests for the stratified metric computation."""

import pytest
import torch

from analysis.metrics.mse import compute_mse
from analysis.metrics.stratified_metric import compute_stratified_metric


@pytest.fixture(name="sample_data")
def sample_data_fixture():
    """Fixture to provide sample data for testing."""
    original_data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    reconstructed_data = torch.tensor([[0.9, 2.1], [3.1, 3.9], [5.1, 5.9], [7.1, 8.1]])
    covariate_data = torch.tensor(
        [[25, 1], [35, 0], [45, 1], [55, 0]]
    )  # Age, Sex (binary categorical)
    return original_data, reconstructed_data, covariate_data


def test_mse_total_single_categorical_covariate(sample_data):
    """Test MSE total with a single categorical covariate (sex)."""
    original_data, reconstructed_data, covariate_data = sample_data
    result = compute_stratified_metric(
        metric_fn=compute_mse,
        original_data=original_data,
        reconstructed_data=reconstructed_data,
        covariate_data=covariate_data,
        covariate_filters=[
            {"index": 1, "values": [1], "type": "categorical"}  # Females (encoded as 1)
        ],
        metric_type="total",
    )
    expected_mse = 0.01  # Expected MSE for samples matching the covariate filter
    assert result == pytest.approx(
        expected_mse, rel=1e-5
    ), f"Expected {expected_mse}, got {result}"


def test_mse_total_single_continuous_covariate(sample_data):
    """Test MSE total with a single continuous covariate (age)."""
    original_data, reconstructed_data, covariate_data = sample_data
    result = compute_stratified_metric(
        metric_fn=compute_mse,
        original_data=original_data,
        reconstructed_data=reconstructed_data,
        covariate_data=covariate_data,
        covariate_filters=[
            {
                "index": 0,
                "values": (30, 50),
                "type": "continuous",
            }  # Age between 30 and 50
        ],
        metric_type="total",
    )
    expected_mse = 0.01  # Expected MSE for samples matching the covariate filter
    assert result == pytest.approx(
        expected_mse, rel=1e-5
    ), f"Expected {expected_mse}, got {result}"


def test_mse_total_multiple_covariates(sample_data):
    """Test MSE total with multiple covariates (age and sex)."""
    original_data, reconstructed_data, covariate_data = sample_data
    result = compute_stratified_metric(
        metric_fn=compute_mse,
        original_data=original_data,
        reconstructed_data=reconstructed_data,
        covariate_data=covariate_data,
        covariate_filters=[
            {
                "index": 0,
                "values": (30, 50),
                "type": "continuous",
            },  # Age between 30 and 50
            {"index": 1, "values": [1], "type": "categorical"},  # Female
        ],
        metric_type="total",
    )
    expected_mse = 0.01  # Manually computed MSE for matching samples
    assert result.item() == pytest.approx(
        expected_mse, rel=1e-5
    ), f"Expected {expected_mse}, got {result}"


def test_mse_sample_multiple_covariates(sample_data):
    """Test MSE per sample with multiple covariates (age and sex)."""
    original_data, reconstructed_data, covariate_data = sample_data
    result = compute_stratified_metric(
        metric_fn=compute_mse,
        original_data=original_data,
        reconstructed_data=reconstructed_data,
        covariate_data=covariate_data,
        covariate_filters=[
            {
                "index": 0,
                "values": (20, 40),
                "type": "continuous",
            },  # Age between 20 and 40
            {"index": 1, "values": [1], "type": "categorical"},  # Female
        ],
        metric_type="sample",
    )
    expected_mse = torch.tensor([0.01])  # Expected MSE per sample for matching samples
    assert torch.allclose(
        result, expected_mse
    ), f"Expected {expected_mse}, got {result}"


def test_empty_filter(sample_data):
    """Test filtering with covariate values that yield no results."""
    original_data, reconstructed_data, covariate_data = sample_data
    with pytest.raises(ValueError, match="Filtering resulted in an empty dataset"):
        compute_stratified_metric(
            metric_fn=compute_mse,
            original_data=original_data,
            reconstructed_data=reconstructed_data,
            covariate_data=covariate_data,
            covariate_filters=[
                {
                    "index": 0,
                    "values": (60, 70),
                    "type": "continuous",
                }  # No samples in this age range
            ],
            metric_type="total",
        )


def test_invalid_covariate_type(sample_data):
    """Test invalid covariate type in covariate_filters."""
    original_data, reconstructed_data, covariate_data = sample_data
    with pytest.raises(
        ValueError, match="Invalid covariate_type or covariate_values format"
    ):
        compute_stratified_metric(
            metric_fn=compute_mse,
            original_data=original_data,
            reconstructed_data=reconstructed_data,
            covariate_data=covariate_data,
            covariate_filters=[
                {
                    "index": 0,
                    "values": (20, 50),
                    "type": "invalid_type",
                }  # Invalid covariate type
            ],
            metric_type="total",
        )


def test_boundary_continuous_filter(sample_data):
    """Test MSE with boundary values for continuous covariate filtering."""
    original_data, reconstructed_data, covariate_data = sample_data
    result = compute_stratified_metric(
        metric_fn=compute_mse,
        original_data=original_data,
        reconstructed_data=reconstructed_data,
        covariate_data=covariate_data,
        covariate_filters=[
            {
                "index": 0,
                "values": (25, 25),
                "type": "continuous",
            }  # Exact match for age 25
        ],
        metric_type="total",
    )
    expected_mse = 0.01  # Expected MSE for exact boundary match
    assert result == pytest.approx(
        expected_mse, rel=1e-5
    ), f"Expected {expected_mse}, got {result}"


def test_invalid_categorical_values_type(sample_data):
    """Test invalid categorical values type (should be a list)."""
    original_data, reconstructed_data, covariate_data = sample_data
    with pytest.raises(
        ValueError, match="For categorical covariates, 'values' should be a list"
    ):
        compute_stratified_metric(
            metric_fn=compute_mse,
            original_data=original_data,
            reconstructed_data=reconstructed_data,
            covariate_data=covariate_data,
            covariate_filters=[
                {
                    "index": 1,
                    "values": 1,
                    "type": "categorical",
                }  # Should be a list, not an integer
            ],
            metric_type="total",
        )
