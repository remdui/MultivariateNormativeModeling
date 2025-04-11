"""Metrics for evaluating the reconstruction performance of the model."""

from typing import Any

from scipy.stats import pearsonr  # type: ignore
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


def calculate_reconstruction_pearson(engine: Any) -> float:
    """
    Calculate the average Pearson correlation coefficient between the original and reconstructed data.

    Each subject's vector of measures is compared (as z-scores) to assess whether
    the relative pattern of values is preserved.
    """
    input_array = get_data_as_numpy(engine.recon_df, "orig", engine.feature_labels)
    recon_array = get_data_as_numpy(engine.recon_df, "recon", engine.feature_labels)

    if input_array is None or recon_array is None:
        engine.logger.warning(
            "Unable to calculate Pearson correlation due to missing data."
        )
        return float("nan")

    correlations = []
    # Calculate the Pearson correlation for each subject (each row)
    for i in range(input_array.shape[0]):
        orig = input_array[i]
        recon = recon_array[i]
        # Ensure enough variability in the values to compute correlation reliably
        if len(set(orig)) < 2 or len(set(recon)) < 2:
            continue
        corr, _ = pearsonr(orig, recon)
        correlations.append(corr)

    if correlations:
        avg_corr = sum(correlations) / len(correlations)
        engine.logger.info(
            f"Average Pearson correlation (across subjects): {avg_corr:.4f}"
        )
        return avg_corr

    engine.logger.warning("No valid subjects for Pearson correlation computation.")
    return float("nan")


def calculate_reconstruction_mse_per_feature(engine: Any) -> dict[str, float]:
    """
    Calculate the mean squared error (MSE) for each feature separately.

    Returns:
        A dictionary with feature names as keys and their corresponding MSE as values.
    """
    input_array = get_data_as_numpy(engine.recon_df, "orig", engine.feature_labels)
    recon_array = get_data_as_numpy(engine.recon_df, "recon", engine.feature_labels)
    if input_array is None or recon_array is None:
        engine.logger.warning(
            "Unable to calculate per-feature MSE due to missing data."
        )
        return {}

    mse_per_feature = {}
    # Iterate over each feature using its column index
    for idx, feature in enumerate(engine.feature_labels):
        mse_value = mean_squared_error(input_array[:, idx], recon_array[:, idx])
        mse_per_feature[feature] = mse_value
        engine.logger.info(
            f"Reconstruction MSE for feature '{feature}': {mse_value:.4f}"
        )

    return mse_per_feature


def calculate_reconstruction_r2_per_feature(engine: Any) -> dict[str, float]:
    """
    Calculate the R² score for each feature separately.

    Returns:
        A dictionary with feature names as keys and their corresponding R² scores as values.
    """
    input_array = get_data_as_numpy(engine.recon_df, "orig", engine.feature_labels)
    recon_array = get_data_as_numpy(engine.recon_df, "recon", engine.feature_labels)
    if input_array is None or recon_array is None:
        engine.logger.warning("Unable to calculate per-feature R² due to missing data.")
        return {}

    r2_per_feature = {}
    # Iterate over each feature using its column index
    for idx, feature in enumerate(engine.feature_labels):
        r2_value = r2_score(input_array[:, idx], recon_array[:, idx])
        r2_per_feature[feature] = r2_value
        engine.logger.info(f"Reconstruction R² for feature '{feature}': {r2_value:.4f}")

    return r2_per_feature
