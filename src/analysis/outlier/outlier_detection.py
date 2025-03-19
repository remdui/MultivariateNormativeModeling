"""Module for outlier detection."""

from typing import Any

import numpy as np

from util.numpy_utils import get_data_as_numpy, get_latent_cols_as_numpy


def detect_outliers(engine: Any, standardized: bool = True) -> dict:
    """Detect outliers in input and reconstruction data."""
    logger = engine.logger
    if not standardized:
        logger.info(
            "Outlier detection is only implemented for standardized data. Skipping."
        )
        return {}

    threshold = engine.properties.data_analysis.features.outlier_threshold

    input_array = get_data_as_numpy(engine.recon_df, "orig", engine.feature_labels)
    recon_array = get_data_as_numpy(engine.recon_df, "recon", engine.feature_labels)
    z_mean_array = get_latent_cols_as_numpy(engine.recon_df, "z_mean_")
    z_varlog_array = get_latent_cols_as_numpy(engine.recon_df, "z_logvar_")

    if input_array is None or recon_array is None:
        logger.warning(
            "Could not detect row-level outliers in input/recon due to missing data."
        )
        row_outlier_input = np.array([], dtype=bool)
        row_outlier_recon = np.array([], dtype=bool)
    else:
        row_outlier_input = (np.abs(input_array) >= threshold).any(axis=1)
        row_outlier_recon = (np.abs(recon_array) >= threshold).any(axis=1)

    if z_mean_array is not None:
        row_outlier_latent_mean = (np.abs(z_mean_array) >= threshold).any(axis=1)
        num_outliers_latent_mean = int(np.sum(row_outlier_latent_mean))
    else:
        num_outliers_latent_mean = 0

    if z_varlog_array is not None:
        row_outlier_latent_varlog = (np.abs(z_varlog_array) >= threshold).any(axis=1)
        num_outliers_latent_varlog = int(np.sum(row_outlier_latent_varlog))
    else:
        num_outliers_latent_varlog = 0

    num_outliers_input = int(np.sum(row_outlier_input))
    num_outliers_recon = int(np.sum(row_outlier_recon))

    if row_outlier_input.shape[0] == row_outlier_recon.shape[0]:
        new_outlier_mask = row_outlier_recon & (~row_outlier_input)
        num_new_outliers = int(np.sum(new_outlier_mask))
    else:
        num_new_outliers = 0

    if row_outlier_input.shape[0] == row_outlier_recon.shape[0]:
        same_outliers_mask = row_outlier_input & row_outlier_recon
        num_same_outliers = int(np.sum(same_outliers_mask))
    else:
        num_same_outliers = 0

    per_feature_input = {}
    per_feature_recon = {}
    per_feature_latent = {}

    if input_array is not None:
        for feature in engine.feature_labels:
            col_name = f"orig_{feature}"
            if col_name in engine.recon_df.columns:
                values = engine.recon_df[col_name].values
                outlier_mask = np.abs(values) >= threshold
                num_out = int(np.sum(outlier_mask))
                if num_out > 0:
                    per_feature_input[feature] = num_out

    if recon_array is not None:
        for feature in engine.feature_labels:
            col_name = f"recon_{feature}"
            if col_name in engine.recon_df.columns:
                values = engine.recon_df[col_name].values
                outlier_mask = np.abs(values) >= threshold
                num_out = int(np.sum(outlier_mask))
                if num_out > 0:
                    per_feature_recon[feature] = num_out

    latent_cols = [
        c
        for c in engine.recon_df.columns
        if c.startswith("z_mean_") or c.startswith("z_logvar_")
    ]
    for col in latent_cols:
        values = engine.recon_df[col].values
        outlier_mask = np.abs(values) >= threshold
        num_out = int(np.sum(outlier_mask))
        if num_out > 0:
            per_feature_latent[col] = num_out

    return {
        "input": num_outliers_input,
        "recon": num_outliers_recon,
        "new_outliers": num_new_outliers,
        "same_outliers": num_same_outliers,
        "latent_mean": num_outliers_latent_mean,
        "latent_varlog": num_outliers_latent_varlog,
        "per_feature_input": per_feature_input,
        "per_feature_recon": per_feature_recon,
        "per_feature_latent": per_feature_latent,
    }


def find_extreme_outliers_in_latent(engine: Any, top_k: int = 5) -> dict:
    """Find extreme outliers in latent space."""
    logger = engine.logger
    if engine.recon_df.empty:
        logger.warning("recon_df is empty. No latent data to analyze.")
        return {}

    z_mean_cols = [col for col in engine.recon_df.columns if col.startswith("z_mean_")]
    if not z_mean_cols:
        logger.warning("No z_mean_* columns found in recon_df. Skipping analysis.")
        return {}

    unique_id_col = engine.properties.dataset.unique_identifier_column
    if unique_id_col not in engine.recon_df.columns:
        logger.error(
            f"Unique identifier column '{unique_id_col}' not found in recon_df."
        )
        return {}

    outlier_dict = {}
    for col in z_mean_cols:
        sorted_df = engine.recon_df[[unique_id_col, col]].sort_values(by=col)
        smallest_outliers = sorted_df.iloc[:top_k][unique_id_col].tolist()
        largest_outliers = sorted_df.iloc[-top_k:][unique_id_col].tolist()
        outlier_dict[col] = {
            "largest_outliers": largest_outliers,
            "smallest_outliers": smallest_outliers,
        }
    return outlier_dict
