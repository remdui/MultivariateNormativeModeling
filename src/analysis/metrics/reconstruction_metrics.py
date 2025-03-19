"""Metrics for evaluating the reconstruction performance of the model."""

from typing import Any

from sklearn.metrics import mean_squared_error, r2_score  # type: ignore

from util.tensor_utils import get_data_as_tensor


def calculate_reconstruction_mse(engine: Any) -> float:
    """Calculate the mean squared error (MSE) between the original and reconstructed data."""
    input_tensor = get_data_as_tensor(engine.recon_df, "orig", engine.feature_labels)
    recon_tensor = get_data_as_tensor(engine.recon_df, "recon", engine.feature_labels)
    if input_tensor is None or recon_tensor is None:
        engine.logger.warning("Unable to calculate MSE due to missing data/tensors.")
        return float("nan")
    mse_value = mean_squared_error(
        input_tensor.cpu().numpy(), recon_tensor.cpu().numpy()
    )
    engine.logger.info(f"Reconstruction MSE (across all features): {mse_value:.4f}")
    return mse_value


def calculate_reconstruction_r2(engine: Any) -> float:
    """Calculate the R² score between the original and reconstructed data."""
    input_tensor = get_data_as_tensor(engine.recon_df, "orig", engine.feature_labels)
    recon_tensor = get_data_as_tensor(engine.recon_df, "recon", engine.feature_labels)
    if input_tensor is None or recon_tensor is None:
        engine.logger.warning("Unable to calculate R² due to missing data/tensors.")
        return float("nan")
    r2_value = r2_score(input_tensor.cpu().numpy(), recon_tensor.cpu().numpy())
    engine.logger.info(f"Reconstruction R² (across all features): {r2_value:.4f}")
    return r2_value
