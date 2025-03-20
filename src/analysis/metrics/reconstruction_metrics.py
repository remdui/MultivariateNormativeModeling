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

    across subjects. Each subject's vector of brain measures is compared (as z-scores) to assess whether
    the relative pattern of values (e.g., which brain regions are higher or lower) is preserved.

    Why it's useful:
    - A high Pearson correlation (close to 1.0) indicates that, even if absolute values differ,
      the overall pattern is well maintained.
    - With z-score normalization, the metric remains scale-invariant.
    - A low or negative correlation suggests that the model is failing to capture the meaningful
      structure in the data.
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
