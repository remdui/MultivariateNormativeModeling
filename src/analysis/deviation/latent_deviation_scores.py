"""Module for computing latent deviation scores."""

import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
from scipy.stats import chi2  # type: ignore


def _get_participant_df(test_df: pd.DataFrame, unique_id_col: str) -> pd.DataFrame:
    """
    Extract and standardize the participant identifier into a column named 'participant_id'.

    from the test DataFrame.
    """
    if unique_id_col in test_df.columns:
        df = test_df[[unique_id_col]].copy()
        df.rename(columns={unique_id_col: "participant_id"}, inplace=True)
    else:
        df = pd.DataFrame({"participant_id": test_df.index})
    return df


def calculate_univariate_deviation_scores(engine: Any) -> pd.DataFrame:
    """
    Compute per-dimension deviation z-scores and percentiles for each test sample.

    Normative statistics (mean and std) are computed from engine.recon_train_df.
    The test sample latent data is taken from engine.recon_df.

    Returns:
        A DataFrame with columns:
          - participant_id (from the unique identifier column defined in properties),
          - {latent_column}_zscore: the z-score computed as (x - μ)/σ,
          - {latent_column}_percentile: the percentile score computed via the standard normal CDF,
          - average_percentile: the average across latent dimensions,
          - average_abs_zscore: average of the absolute z-scores across latent dimensions.
    """
    logger = engine.logger

    normative_df = engine.recon_train_df
    test_df = engine.recon_df
    unique_id_col = engine.properties.dataset.unique_identifier_column

    # Identify latent columns (assuming they start with "z_mean_")
    latent_cols = [col for col in normative_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent columns found with prefix 'z_mean_' in normative data.")
        return pd.DataFrame()

    # Compute normative statistics for each latent dimension.
    norm_means = normative_df[latent_cols].mean()
    norm_stds = normative_df[latent_cols].std()

    # Get the participant DataFrame using the unique identifier column.
    results_df = _get_participant_df(test_df, unique_id_col)

    percentile_list = []
    abs_zscore_list = []

    for col in latent_cols:
        mean_val = norm_means[col]
        std_val = norm_stds[col]

        if std_val == 0:
            z_values = np.full(shape=test_df.shape[0], fill_value=0.0)
            percentiles = np.full(shape=test_df.shape[0], fill_value=50.0)
        else:
            z_values = (test_df[col] - mean_val) / std_val
            percentiles = stats.norm.cdf(z_values) * 100

        z_arr = np.asarray(z_values)
        pct_arr = np.asarray(percentiles)

        results_df[f"{col}_zscore"] = z_arr
        results_df[f"{col}_percentile"] = pct_arr

        percentile_list.append(pct_arr)
        abs_zscore_list.append(np.abs(z_arr))

    results_df["average_percentile"] = np.mean(np.column_stack(percentile_list), axis=1)
    results_df["average_abs_zscore"] = np.mean(np.column_stack(abs_zscore_list), axis=1)

    logger.info(
        "Computed univariate deviation scores (z-scores and percentiles) for each test sample."
    )
    return results_df


def calculate_mahalanobis_deviation_scores(engine: Any) -> pd.DataFrame:
    """
    Compute a multivariate deviation score using the Mahalanobis distance.

    Normative parameters (mean vector and covariance matrix) are computed from engine.recon_train_df.
    For each test sample in engine.recon_df, the Mahalanobis distance is computed and then
    converted to a percentile using the chi-square distribution (degrees of freedom = number of latent dimensions).

    Returns:
        A DataFrame with columns:
          - participant_id (from the unique identifier column),
          - mahalanobis_distance: the computed Mahalanobis distance,
          - mahalanobis_percentile: the percentile (0-100) of that distance.
    """
    logger = engine.logger

    normative_df = engine.recon_train_df
    test_df = engine.recon_df
    unique_id_col = engine.properties.dataset.unique_identifier_column

    latent_cols = [col for col in normative_df.columns if col.startswith("z_mean_")]
    if not latent_cols:
        logger.error("No latent columns found with prefix 'z_mean_' in normative data.")
        return pd.DataFrame()

    norm_data = normative_df[latent_cols].to_numpy()
    mean_vector = np.mean(norm_data, axis=0)
    cov_matrix = np.cov(norm_data, rowvar=False)

    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        logger.error(
            "Covariance matrix is singular; cannot compute Mahalanobis distance."
        )
        return pd.DataFrame()

    results_df = _get_participant_df(test_df, unique_id_col)
    test_data = test_df[latent_cols].to_numpy()
    distances = []
    for x in test_data:
        diff = x - mean_vector
        dist_sq = np.dot(np.dot(diff, inv_cov_matrix), diff.T)
        distances.append(np.sqrt(dist_sq))
    distances = np.array(distances)
    results_df["mahalanobis_distance"] = distances

    p = len(latent_cols)
    results_df["mahalanobis_percentile"] = chi2.cdf(distances**2, df=p) * 100

    logger.info("Computed Mahalanobis deviation scores for each test sample.")
    return results_df


def save_deviation_scores_to_csv(
    engine: Any, univariate_df: pd.DataFrame, mahalanobis_df: pd.DataFrame
) -> None:
    """
    Merge the univariate and Mahalanobis deviation scores and save to a CSV file,.

    if available. If either DataFrame is empty, use the one that is non-empty.
    If both are empty, do nothing.

    The CSV is saved to:
      os.path.join(output_dir, "metrics", f"{output_identifier}.csv")

    Args:
        engine (Any): The analysis engine instance containing properties such as output_dir and unique identifier.
        univariate_df (pd.DataFrame): DataFrame with univariate deviation scores.
        mahalanobis_df (pd.DataFrame): DataFrame with Mahalanobis deviation scores.
    """
    logger = engine.logger

    if univariate_df.empty and mahalanobis_df.empty:
        logger.error(
            "Both univariate and Mahalanobis deviation score DataFrames are empty. Nothing to merge or save."
        )
        return
    if univariate_df.empty:
        logger.warning(
            "Univariate deviation scores DataFrame is empty. Using Mahalanobis deviation scores only."
        )
        merged_df = mahalanobis_df.copy()
    elif mahalanobis_df.empty:
        logger.warning(
            "Mahalanobis deviation scores DataFrame is empty. Using univariate deviation scores only."
        )
        merged_df = univariate_df.copy()
    else:
        merged_df = pd.merge(
            univariate_df, mahalanobis_df, on="participant_id", how="outer"
        )

    output_dir = engine.properties.system.output_dir
    output_identifier = "deviation_scores"
    filename = os.path.join(output_dir, "metrics", f"{output_identifier}.csv")

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    merged_df.to_csv(filename, index=False)
    logger.info(f"Saved deviation scores to CSV: {filename}")
