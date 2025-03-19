"""Metrics for evaluating the reconstruction performance of the model."""

from typing import Any

from sklearn.metrics import mean_squared_error, r2_score  # type: ignore

from util.numpy_utils import get_data_as_numpy


def calculate_reconstruction_mse(engine: Any) -> float:
    """Calculate the mean squared error (MSE) between the original and reconstructed data."""
    input_array = get_data_as_numpy(engine.recon_df, "orig", engine.feature_labels)
    recon_array = get_data_as_numpy(engine.recon_df, "recon", engine.feature_labels)
    if input_array is None or recon_array is None:
        engine.logger.warning("Unable to calculate MSE due to missing data.")
        return float("nan")
    mse_value = mean_squared_error(input_array, recon_array)
    engine.logger.info(f"Reconstruction MSE (across all features): {mse_value:.4f}")
    return mse_value


def calculate_reconstruction_r2(engine: Any) -> float:
    """Calculate the R² score between the original and reconstructed data."""
    input_array = get_data_as_numpy(engine.recon_df, "orig", engine.feature_labels)
    recon_array = get_data_as_numpy(engine.recon_df, "recon", engine.feature_labels)
    if input_array is None or recon_array is None:
        engine.logger.warning("Unable to calculate R² due to missing data.")
        return float("nan")
    r2_value = r2_score(input_array, recon_array)
    engine.logger.info(f"Reconstruction R² (across all features): {r2_value:.4f}")
    return r2_value
