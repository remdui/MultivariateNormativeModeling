"""Metrics for evaluating the reconstruction performance of the model."""

import logging
from math import exp, sqrt
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, pearsonr  # type: ignore
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


def compare_feature_distributions_ks(
    engine: Any,
) -> Any:
    """
    Compare the distributions of original and reconstructed features using the.

    two-sample Kolmogorov–Smirnov (KS) test for each feature. For each feature,
    the function computes the KS statistic and p-value. A 'similarity' score is
    computed as 1 - KS_statistic, where 1 indicates identical distributions.

    The function returns a dictionary with keys being the feature names and values
    being dictionaries containing:
      - ks_statistic: KS statistic between the original and reconstructed distributions.
      - p_value: Two-sided p-value from the KS test.
      - similarity: Computed as 1 - ks_statistic.

    An additional key 'average_similarity' is added to indicate the average similarity
    across all features.

    Args:
        engine (Any): Analysis engine instance with a loaded `recon_df` that contains
                      columns with names like "orig_feature" and "recon_feature".

    Returns:
        dict[str, float | dict[str, float]]: A dictionary containing per-feature metrics and
                                              overall average similarity.
    """
    logger = engine.logger if hasattr(engine, "logger") else logging.getLogger(__name__)

    if engine.recon_df.empty:
        logger.warning("recon_df is empty. Cannot compare distributions.")
        return {}

    feature_names = [
        col[5:] for col in engine.recon_df.columns if col.startswith("orig_")
    ]
    results = {}
    similarities = []

    for feature in feature_names:
        orig_col = f"orig_{feature}"
        recon_col = f"recon_{feature}"
        if orig_col in engine.recon_df.columns and recon_col in engine.recon_df.columns:
            data_orig = engine.recon_df[orig_col].dropna()
            data_recon = engine.recon_df[recon_col].dropna()

            if len(data_orig) > 0 and len(data_recon) > 0:
                ks_stat, p_value = ks_2samp(data_orig, data_recon)
                similarity = (
                    1 - ks_stat
                )  # 1 means identical, 0 means completely dissimilar.
            else:
                ks_stat, p_value, similarity = None, None, None

            results[feature] = {
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "similarity": similarity,
            }
            if similarity is not None:
                similarities.append(similarity)
            logger.info(
                f"Feature '{feature}': KS statistic = {ks_stat:.4f}, p-value = {p_value:.4f}, similarity = {similarity:.4f}"
            )
        else:
            logger.warning(
                f"Either {orig_col} or {recon_col} not found in recon_df. Skipping feature '{feature}'."
            )

    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        results["average_similarity"] = avg_similarity
        logger.info(f"Average similarity across all features: {avg_similarity:.4f}")
    else:
        results["average_similarity"] = None
        logger.warning("No valid features were compared; average similarity is None.")

    return results


def compare_feature_distributions_bhattacharyya(
    engine: Any,
) -> Any:
    """
    Compare the distributions of original and reconstructed features assuming both are Gaussian.

    For each feature, the function estimates the mean and standard deviation for both the original
    and reconstructed distributions. The Bhattacharyya coefficient (BC) is then computed as:

        BC = sqrt((2 * σ_orig * σ_recon) / (σ_orig² + σ_recon²)) * exp(-((μ_orig - μ_recon)²) / (4*(σ_orig² + σ_recon²)))

    A BC of 1 indicates identical distributions, while lower values indicate increasing dissimilarity.
    The function returns a dictionary with keys being the feature names and values being dictionaries containing:
      - mu_orig: Mean of the original distribution.
      - sigma_orig: Standard deviation of the original distribution.
      - mu_recon: Mean of the reconstructed distribution.
      - sigma_recon: Standard deviation of the reconstructed distribution.
      - bhattacharyya_coefficient: The computed Bhattacharyya coefficient.

    An additional key 'average_similarity' is added to indicate the average Bhattacharyya coefficient across all features.

    Args:
        engine (Any): Analysis engine instance with a loaded `recon_df` that contains columns with names
                      like "orig_feature" and "recon_feature".

    Returns:
        dict[str, float | dict[str, float]]: A dictionary containing per-feature metrics and overall average similarity.
    """
    logger = engine.logger if hasattr(engine, "logger") else logging.getLogger(__name__)

    if engine.recon_df.empty:
        logger.warning("recon_df is empty. Cannot compare distributions.")
        return {}

    feature_names = [
        col[5:] for col in engine.recon_df.columns if col.startswith("orig_")
    ]
    results = {}
    similarities = []

    for feature in feature_names:
        orig_col = f"orig_{feature}"
        recon_col = f"recon_{feature}"
        if orig_col in engine.recon_df.columns and recon_col in engine.recon_df.columns:
            data_orig = engine.recon_df[orig_col].dropna().to_numpy()
            data_recon = engine.recon_df[recon_col].dropna().to_numpy()

            if len(data_orig) > 0 and len(data_recon) > 0:
                mu_orig = np.mean(data_orig)
                sigma_orig = np.std(data_orig)
                mu_recon = np.mean(data_recon)
                sigma_recon = np.std(data_recon)

                # Prevent division by zero when both variances are 0
                denominator = sigma_orig**2 + sigma_recon**2
                if denominator == 0:
                    bc = 1.0
                else:
                    bc = sqrt((2 * sigma_orig * sigma_recon) / denominator) * exp(
                        -((mu_orig - mu_recon) ** 2) / (4 * denominator)
                    )
            else:
                mu_orig, sigma_orig, mu_recon, sigma_recon, bc = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )

            results[feature] = {
                "mu_orig": mu_orig,
                "sigma_orig": sigma_orig,
                "mu_recon": mu_recon,
                "sigma_recon": sigma_recon,
                "bhattacharyya_coefficient": bc,
            }
            if bc is not None:
                similarities.append(bc)
            logger.info(
                f"Feature '{feature}': mu_orig = {mu_orig:.4f}, sigma_orig = {sigma_orig:.4f}, "
                f"mu_recon = {mu_recon:.4f}, sigma_recon = {sigma_recon:.4f}, BC = {bc:.4f}"
            )
        else:
            logger.warning(
                f"Either {orig_col} or {recon_col} not found in recon_df. Skipping feature '{feature}'."
            )

    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        results["average_similarity"] = avg_similarity
        logger.info(
            f"Average Bhattacharyya similarity across all features: {avg_similarity:.4f}"
        )
    else:
        results["average_similarity"] = None
        logger.warning("No valid features were compared; average similarity is None.")

    return results
